"""Group Relative Policy Optimization (Shao et al., 2024 — DeepSeekMath; used for DeepSeek-R1), written by hand.

For each prompt q, sample a group of G completions o_1..o_G from the current policy, score each with a verifiable reward
r_i (here: execute the SQL, compare with the gold result), and use the group itself as the baseline:

    A_i = (r_i − mean(r)) / (std(r) + ε)                         every token of o_i gets advantage A_i
    ratio_t = π_θ(o_t | q, o_<t) / π_old(o_t | q, o_<t)
    L = − mean_t [ min(ratio_t · A_i, clip(ratio_t, 1 − ε_c, 1 + ε_c) · A_i) − β · KL_t ]
    KL_t = π_ref/π_θ − log(π_ref/π_θ) − 1                         (the k3 estimator: unbiased, always ≥ 0)

No value network (that is the point of GRPO). π_ref is the SFT model: with LoRA it is the same network with the
adapter switched off, so the reference costs no extra memory. With one optimisation pass per batch (μ = 1) π_old = π_θ
and the ratio is exactly 1; the clipped form only matters for μ > 1. Groups whose rewards are all equal carry no
signal (A = 0); their share is logged — it is the first number to watch.

Token layout for the log-prob pass: [left pad][prompt][completion][right pad], so the completion always occupies the
last C positions and only those logits are materialised (`logits_to_keep`), which keeps memory flat in the prompt length.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import GRPO, SEED, GRPORules


@dataclass
class Samples:
    seq: torch.Tensor  # (N, P + C) token ids  # noqa: F821
    attn: torch.Tensor  # (N, P + C)  # noqa: F821
    pos: torch.Tensor  # (N, P + C) position ids  # noqa: F821
    comp_mask: torch.Tensor  # (N, C) 1 on real completion tokens  # noqa: F821
    texts: list[str]
    group: np.ndarray  # (N,) prompt index of each sample
    C: int
    temperature: float = 1.0


def sample_groups(model, tok, prompts: list[str], G: int, temperature: float, max_new_tokens: int, device) -> Samples:
    import torch

    model.eval()
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=max_new_tokens, num_return_sequences=G,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    P = enc["input_ids"].shape[1]
    comp = out[:, P:]
    eos = tok.eos_token_id
    # keep tokens up to and including the first EOS; everything after is padding
    is_eos = comp == eos
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((comp.shape[0],), comp.shape[1], device=comp.device))
    ar = torch.arange(comp.shape[1], device=comp.device)[None, :]
    comp_mask = (ar <= first[:, None]).long()  # through the first EOS; all positions when the length cap was hit
    C = int(comp_mask.sum(1).max().item()) if comp_mask.numel() else 0
    C = max(C, 1)
    comp, comp_mask = comp[:, :C], comp_mask[:, :C]
    comp = torch.where(comp_mask.bool(), comp, torch.full_like(comp, tok.pad_token_id))
    p_ids = enc["input_ids"].repeat_interleave(G, 0)
    p_mask = enc["attention_mask"].repeat_interleave(G, 0)
    seq = torch.cat([p_ids, comp], 1)
    attn = torch.cat([p_mask, comp_mask], 1)
    pos = (attn.cumsum(1) - 1).clamp(min=0)
    texts = [tok.decode(comp[i][comp_mask[i].bool()], skip_special_tokens=True) for i in range(comp.shape[0])]
    return Samples(seq, attn, pos, comp_mask, texts, np.repeat(np.arange(len(prompts)), G), C, temperature)


def token_logprobs(model, s: Samples, idx, use_amp: bool):
    """Log-probabilities of the completion tokens for the samples in `idx`: (len(idx), C)."""
    import torch

    seq, attn, pos = s.seq[idx], s.attn[idx], s.pos[idx]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(input_ids=seq, attention_mask=attn, position_ids=pos, logits_to_keep=s.C + 1)
    logits = out.logits[:, :-1].float() / s.temperature  # position P−1+t predicts completion token t; same temperature as sampling
    tgt = seq[:, -s.C :]
    return torch.log_softmax(logits, -1).gather(-1, tgt[..., None]).squeeze(-1)


def advantages(rewards: np.ndarray, group: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, float]:
    """Group-normalised advantages and the share of groups with zero reward variance."""
    adv = np.zeros_like(rewards, dtype=float)
    zero = 0
    groups = np.unique(group)
    for g in groups:
        m = group == g
        r = rewards[m]
        sd = r.std()
        if sd < 1e-8:
            zero += 1
            continue
        adv[m] = (r - r.mean()) / (sd + eps)
    return adv, zero / max(len(groups), 1)


def grpo_loss(lp, old_lp, ref_lp, adv, mask, beta: float, clip_eps: float):
    """Per-token GRPO objective (to be minimised) and the per-token KL, both (n, C)."""
    import torch

    old = lp.detach() if old_lp is None else old_lp
    ratio = torch.exp(lp - old)
    a = adv[:, None]
    pg = -torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    if ref_lp is not None and beta > 0:
        d = ref_lp - lp
        kl = torch.exp(d) - d - 1
    else:
        kl = torch.zeros_like(lp)
    return (pg + beta * kl) * mask, kl * mask


def grpo_step(model, tok, prompts: list[str], reward_fn: Callable[[int, str], tuple[float, dict]], opt, rules: GRPORules, device: str, ref: bool = True) -> dict:
    """One GRPO update. reward_fn(prompt_index, completion_text) → (reward, info dict with optional 'correct', 'executed', 'flags')."""
    import torch

    use_amp = device == "cuda"
    t0 = time.perf_counter()
    s = sample_groups(model, tok, prompts, rules.group_size, rules.temperature, rules.max_new_tokens, device)
    t_gen = time.perf_counter() - t0
    infos = [reward_fn(int(g), t) for g, t in zip(s.group, s.texts)]
    rewards = np.array([r for r, _ in infos], dtype=float)
    adv_np, zero_share = advantages(rewards, s.group)
    adv = torch.tensor(adv_np, dtype=torch.float32, device=s.seq.device)
    mask = s.comp_mask.float()
    n = s.seq.shape[0]
    tokens_total = float(mask.sum().item())
    mb = max(1, rules.micro_batch)
    # reference and (for μ > 1) old log-probs, no grad
    ref_lp = old_lp = None
    with torch.no_grad():
        if ref and rules.beta > 0 and hasattr(model, "disable_adapter"):
            parts = []
            with model.disable_adapter():
                for i in range(0, n, mb):
                    parts.append(token_logprobs(model, s, slice(i, i + mb), use_amp))
            ref_lp = torch.cat(parts)
        if rules.inner_epochs > 1:
            old_lp = torch.cat([token_logprobs(model, s, slice(i, i + mb), use_amp) for i in range(0, n, mb)])
    model.train()
    t1 = time.perf_counter()
    kl_sum, loss_sum = 0.0, 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    for _ in range(rules.inner_epochs):
        opt.zero_grad(set_to_none=True)
        for i in range(0, n, mb):
            sl = slice(i, i + mb)
            lp = token_logprobs(model, s, sl, use_amp)
            per_tok, kl = grpo_loss(lp, None if old_lp is None else old_lp[sl], None if ref_lp is None else ref_lp[sl], adv[sl], mask[sl], rules.beta, rules.clip_eps)
            if rules.loss_agg == "token":
                loss = per_tok.sum() / max(tokens_total, 1.0)
            else:
                loss = (per_tok.sum(1) / mask[sl].sum(1).clamp(min=1)).sum() / n
            loss.backward()
            loss_sum += float(loss.detach())
            kl_sum += float(kl.detach().sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    t_train = time.perf_counter() - t1
    flags: dict[str, int] = {}
    for _, info in infos:
        for f in info.get("flags", ()):
            flags[f] = flags.get(f, 0) + 1
    return {"reward": float(rewards.mean()), "correct": float(np.mean([i.get("correct", False) for _, i in infos])),
            "executed": float(np.mean([i.get("executed", False) for _, i in infos])), "zero_var_groups": zero_share, "loss": loss_sum / rules.inner_epochs,
            "kl": kl_sum / max(tokens_total, 1.0) / rules.inner_epochs, "completion_tokens": tokens_total / n,
            "flags": {k: v / n for k, v in flags.items()}, "gen_s": round(t_gen, 2), "train_s": round(t_train, 2)}


def train_grpo(model, tok, prompt_pool: list[str], reward_fn: Callable[[int, str], tuple[float, dict]], out_dir: Path, device: str,
               rules: GRPORules = GRPO, seed: int = SEED, eval_fn: Callable[[], float] | None = None, log=print, steps: int | None = None) -> dict:
    """The loop: each step draws `prompts_per_step` prompts from the pool; eval_fn() (greedy val EX) every eval_every
    steps; the adapter with the best eval score is saved (or the last one when there is no eval_fn)."""
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rules.lr, weight_decay=0.0)
    total = steps or rules.steps
    history, best, t0 = [], -1.0, time.time()
    for step in range(1, total + 1):
        idx = rng.choice(len(prompt_pool), size=min(rules.prompts_per_step, len(prompt_pool)), replace=False)
        stats = grpo_step(model, tok, [prompt_pool[i] for i in idx], lambda g, text, idx=idx: reward_fn(int(idx[g]), text), opt, rules, device)
        stats["step"] = step
        if eval_fn is not None and (step % rules.eval_every == 0 or step == total):
            stats["val_ex"] = float(eval_fn())
            if stats["val_ex"] > best:
                best = stats["val_ex"]
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
        history.append(stats)
        if log and (step % 10 == 0 or step == 1 or "val_ex" in stats):
            log(f"step {step}/{total}  reward {stats['reward']:.3f}  correct {stats['correct']:.3f}  zero-var {stats['zero_var_groups']:.2f}  "
                f"kl {stats['kl']:.4f}  len {stats['completion_tokens']:.0f}  gen {stats['gen_s']}s train {stats['train_s']}s"
                + (f"  val EX {stats['val_ex']:.4f}" if "val_ex" in stats else ""))
    if eval_fn is None:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    info = {"steps": total, "best_val_ex": best, "minutes": round((time.time() - t0) / 60, 1), "rules": asdict(rules), "history": history}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "grpo_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
