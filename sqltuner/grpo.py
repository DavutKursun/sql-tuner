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
    truncated: np.ndarray | None = None  # (N,) no end-of-turn token within max_new_tokens


def sample_groups(model, tok, prompts: list[str], G: int, temperature: float, max_new_tokens: int, device, max_cells: int = 8 * 2048**2) -> Samples:
    """G completions per prompt. generate() copies each prompt G times before the prefill, whose attention is quadratic in
    the prompt length, so whole prompt groups are generated in chunks keeping sequences x prompt_length² under `max_cells`
    (a T4 ran out of memory prefilling 32 sequences of ~1.7k tokens at once). Every chunk is a slice of the same
    left-padded batch, so prompts are tokenised and positioned exactly as in a single call."""
    import torch

    from .generate import sampling_kwargs

    model.eval()
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    P = enc["input_ids"].shape[1]
    per_chunk = max(1, max_cells // (G * P * P))
    parts = []
    with torch.no_grad():  # plain sampling at `temperature` (no top-k / top-p / repetition penalty): token_logprobs scores the same distribution
        for c in range(0, len(prompts), per_chunk):
            out = model.generate(input_ids=enc["input_ids"][c : c + per_chunk], attention_mask=enc["attention_mask"][c : c + per_chunk],
                                 **sampling_kwargs(temperature), max_new_tokens=max_new_tokens, num_return_sequences=G, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
            parts.append(out[:, P:])
    if device == "cuda":
        torch.cuda.empty_cache()
    width = max(p.shape[1] for p in parts)
    comp = torch.cat([torch.nn.functional.pad(p, (0, width - p.shape[1]), value=tok.pad_token_id) for p in parts])
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
    truncated = (~is_eos.any(1)).cpu().numpy()
    return Samples(seq, attn, pos, comp_mask, texts, np.repeat(np.arange(len(prompts)), G), C, temperature, truncated)


def token_logprobs(model, s: Samples, idx, amp=None):
    """Log-probabilities of the completion tokens for the samples in `idx`: (len(idx), C). `amp` = autocast dtype or None."""
    import torch

    seq, attn, pos = s.seq[idx], s.attn[idx], s.pos[idx]
    with torch.autocast(device_type="cuda", dtype=amp or torch.float16, enabled=bool(amp)):
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




HACK_FLAGS = ("empty_result", "added_limit", "select_star", "row_cap", "row_flood", "truncated")


def grpo_step(model, tok, prompts: list[str], reward_fn: Callable[[int, str], tuple[float, dict]], opt, rules: GRPORules, device: str, ref: bool = True,
              scaler=None) -> dict:
    """One GRPO update. reward_fn(prompt_index, completion_text) → (reward, info with optional 'correct', 'executed',
    'flags', 'failure', 'sql'). A completion that reaches max_new_tokens without its end-of-turn token earns 0 whatever it
    contains (flag 'truncated'). `scaler`: the run's GradScaler (fp16 autocast, i.e. a T4; train_grpo keeps one).
    Returns the step's monitors; `samples` lists the flagged completions (batch index, text, reward, info)."""
    import torch

    from .generate import amp_dtype

    amp = amp_dtype(device)  # autocast dtype or None
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=False)
    cuda = device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    s = sample_groups(model, tok, prompts, rules.group_size, rules.temperature, rules.max_new_tokens, device)
    t_gen = time.perf_counter() - t0
    cut = s.truncated if s.truncated is not None else np.zeros(len(s.texts), dtype=bool)
    infos = []
    for g, text, c in zip(s.group, s.texts, cut):
        r, info = reward_fn(int(g), text)
        if c:  # never finished: no credit and not counted as executed, whatever the text holds
            r, info = 0.0, {**info, "correct": False, "executed": False, "failure": None, "flags": (*info.get("flags", ()), "truncated")}
        infos.append((r, info))
    t_reward = time.perf_counter() - t0 - t_gen
    rewards = np.array([r for r, _ in infos], dtype=float)
    adv_np, zero_share = advantages(rewards, s.group)
    adv = torch.tensor(adv_np, dtype=torch.float32, device=s.seq.device)
    mask = s.comp_mask.float()
    n = s.seq.shape[0]
    tokens_total = float(mask.sum().item())
    mb = max(1, rules.micro_batch)
    # reference and (for μ > 1) old log-probs, no grad; with μ = 1 π_old is π_θ detached, so no extra forward
    t1 = time.perf_counter()
    ref_lp = old_lp = None
    with torch.no_grad():
        if ref and rules.beta > 0 and hasattr(model, "disable_adapter"):
            parts = []
            with model.disable_adapter():
                for i in range(0, n, mb):
                    parts.append(token_logprobs(model, s, slice(i, i + mb), amp))
            ref_lp = torch.cat(parts)
        if rules.inner_epochs > 1:
            old_lp = torch.cat([token_logprobs(model, s, slice(i, i + mb), amp) for i in range(0, n, mb)])
    t_ref = time.perf_counter() - t1
    model.train()
    t2 = time.perf_counter()
    kl_sum, loss_sum = 0.0, 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    for _ in range(rules.inner_epochs):
        opt.zero_grad(set_to_none=True)
        for i in range(0, n, mb):
            sl = slice(i, i + mb)
            lp = token_logprobs(model, s, sl, amp)
            per_tok, kl = grpo_loss(lp, None if old_lp is None else old_lp[sl], None if ref_lp is None else ref_lp[sl], adv[sl], mask[sl], rules.beta, rules.clip_eps)
            if rules.loss_agg == "token":
                loss = per_tok.sum() / max(tokens_total, 1.0)
            else:
                loss = (per_tok.sum(1) / mask[sl].sum(1).clamp(min=1)).sum() / n
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            kl_sum += float(kl.detach().sum())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
    t_train = time.perf_counter() - t2
    if cuda:
        torch.cuda.empty_cache()
    lengths = s.comp_mask.sum(1).cpu().numpy()
    correct = np.array([bool(i.get("correct")) for _, i in infos])
    executed = np.array([bool(i.get("executed")) for _, i in infos])
    flags: dict[str, int] = {}
    failures: dict[str, int] = {}
    for _, info in infos:
        for f in info.get("flags", ()):
            flags[f] = flags.get(f, 0) + 1
        if info.get("failure"):
            failures[info["failure"]] = failures.get(info["failure"], 0) + 1
    samples = [{"i": j, "group": int(s.group[j]), "text": s.texts[j], "reward": r, "correct": bool(info.get("correct")), "executed": bool(info.get("executed")),
                "flags": list(info.get("flags", ())), "sql": info.get("sql")} for j, (r, info) in enumerate(infos) if set(info.get("flags", ())) & set(HACK_FLAGS)]
    return {"reward": float(rewards.mean()), "correct": float(correct.mean()), "executed": float(executed.mean()), "executed_wrong": float((executed & ~correct).mean()),
            "failures": {k: v / n for k, v in sorted(failures.items())}, "zero_var_groups": zero_share, "loss": loss_sum / rules.inner_epochs,
            "kl": kl_sum / max(tokens_total, 1.0) / rules.inner_epochs, "completion_tokens": float(lengths.mean()),
            "completion_tokens_p95": float(np.percentile(lengths, 95)), "truncated": float(cut.mean()), "flags": {k: v / n for k, v in sorted(flags.items())},
            "gen_s": round(t_gen, 2), "reward_s": round(t_reward, 2), "ref_s": round(t_ref, 2), "train_s": round(t_train, 2),
            "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if cuda else None, "samples": samples}


def prompt_order(n: int, steps: int, k: int, p: np.ndarray | None = None, seed: int = SEED) -> list[list[int]]:
    """The pool indices of every step, drawn up front from `seed` (with probabilities `p` over the pool, else uniform), so
    a run and its ablation see the same prompts in the same order."""
    rng = np.random.default_rng(seed)
    return [[int(i) for i in rng.choice(n, size=min(k, n), replace=False, p=p)] for _ in range(steps)]


def train_grpo(model, tok, prompt_pool: list[str], reward_fn: Callable[[int, str], tuple[float, dict]], out_dir: Path, device: str,
               rules: GRPORules = GRPO, seed: int = SEED, eval_fn: Callable[[int], float] | None = None, log=print, steps: int | None = None,
               weights: np.ndarray | None = None, max_hours: float | None = None, prompt_ids: list[str] | None = None, keep_examples: int = 6) -> dict:
    """The loop. Every step's prompts are drawn up front (`prompt_order`, saved to prompt_order.json with its hash). At
    step 0 and every eval_every steps the adapter is saved to out_dir/step_NNN and eval_fn(step) runs (greedy EX on the
    val subset); the best score (ties: the earlier step) is also saved in out_dir itself. `max_hours` stops the loop
    before a step that would leave too little time for a last checkpoint and eval. Up to `keep_examples` completions
    are kept per hacking flag."""
    import hashlib

    import torch

    from .generate import amp_dtype

    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rules.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(device) == torch.float16)  # fp16 (T4) needs loss scaling; bf16 does not
    total = steps or rules.steps
    order = prompt_order(len(prompt_pool), total, rules.prompts_per_step, weights, seed)
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:16]
    (out_dir / "prompt_order.json").write_text(json.dumps({"hash": order_hash, "steps": order,
                                                           "ids": [[prompt_ids[i] for i in o] for o in order] if prompt_ids else None}), encoding="utf-8")
    history: list[dict] = []
    evals: list[dict] = []
    examples: dict[str, list[dict]] = {}
    state = {"best": -1.0, "best_step": None, "eval_s": 0.0}
    t0 = time.time()

    def checkpoint(step: int) -> float | None:
        model.save_pretrained(out_dir / f"step_{step:03d}")
        if eval_fn is None:
            return None
        te = time.time()
        ex = float(eval_fn(step))
        state["eval_s"] = time.time() - te
        evals.append({"step": step, "val_ex": ex, "minutes": round((time.time() - t0) / 60, 1), "eval_s": round(state["eval_s"], 1)})
        if ex > state["best"]:  # strictly better: a tie keeps the earlier step
            state["best"], state["best_step"] = ex, step
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        if log:
            log(f"step {step}: val-subset EX {ex:.4f} ({state['eval_s']:.0f} s)")
        return ex

    checkpoint(0)
    stopped, done = None, 0
    for step in range(1, total + 1):
        if max_hours and history:
            train_s = time.time() - t0 - sum(e["eval_s"] for e in evals)
            if time.time() - t0 + train_s / len(history) + 1.2 * state["eval_s"] + 60 > max_hours * 3600:
                stopped = f"wall-clock guard at {max_hours} h"
                break
        idx = order[step - 1]
        stats = grpo_step(model, tok, [prompt_pool[i] for i in idx], lambda g, text, idx=idx: reward_fn(int(idx[g]), text), opt, rules, device, scaler=scaler)
        for smp in stats.pop("samples"):
            for f in smp["flags"]:
                if f in HACK_FLAGS and len(examples.setdefault(f, [])) < keep_examples:
                    pid = idx[smp["group"]]
                    examples[f].append({"step": step, "prompt": prompt_ids[pid] if prompt_ids else pid, "sql": smp["sql"], "reward": smp["reward"],
                                        "correct": smp["correct"], "executed": smp["executed"], "flags": smp["flags"]})
        stats.update(step=step, minutes=round((time.time() - t0) / 60, 2))
        history.append(stats)
        done = step
        if step % rules.eval_every == 0 or step == total:
            stats["val_ex"] = checkpoint(step)
        if log and (step % 10 == 0 or step == 1):
            log(f"step {step}/{total}  reward {stats['reward']:.3f}  correct {stats['correct']:.3f}  exec-wrong {stats['executed_wrong']:.3f}  "
                f"zero-var {stats['zero_var_groups']:.2f}  kl {stats['kl']:.4f}  len {stats['completion_tokens']:.0f}  trunc {stats['truncated']:.2f}  "
                f"gen {stats['gen_s']}s ref {stats['ref_s']}s train {stats['train_s']}s")
    if stopped:
        if log:
            log(f"{stopped}: stopping after step {done}")
        if done and (not evals or evals[-1]["step"] != done):
            checkpoint(done)  # the last state, saved and evaluated
    if eval_fn is None:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    info = {"steps": done, "planned_steps": total, "stopped": stopped, "best_val_ex": state["best"], "best_step": state["best_step"], "evals": evals,
            "minutes": round((time.time() - t0) / 60, 1), "order_hash": order_hash, "rules": asdict(rules), "examples": examples, "history": history}
    (out_dir / "grpo_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
