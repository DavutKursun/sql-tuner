"""Supervised fine-tuning with LoRA, written as a plain PyTorch loop (no Trainer), so every step is visible:

    prompt = chat template(system, schema + question) with the generation prompt
    target = "```sql\n<gold>\n```" + end-of-turn token
    loss   = cross-entropy on the target tokens only (prompt tokens get label −100)

On CUDA bf16 autocast where the GPU has it natively (A100, L4, H100), else fp16 autocast + GradScaler (T4); fp32 on
CPU; gradient checkpointing; gradient accumulation; cosine schedule with warmup; the adapter with the lowest validation
loss is kept. Logits are computed only at the positions that predict a target token: the prompt (schema + question)
is up to 97% of a sequence, and full fp32 logits over a 151,936-word vocabulary would cost ~2.5 GB per 4,096 tokens.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import LORA, PROMPT, SEED, SFT, LoraRules, PromptRules, SFTRules
from .data import Example, db_path
from .prompts import apply_chat_template, messages, schema_text, target

IGNORE = -100


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]


def encode(tok, msgs: list[dict], answer: str) -> Encoded:
    """Prompt and answer are tokenized separately and concatenated, so the label boundary is exact."""
    prompt = apply_chat_template(tok, msgs, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    return Encoded(p_ids + a_ids, [IGNORE] * len(p_ids) + a_ids)


def build_examples(examples: list[Example], tok, spider_dir: Path, rules: PromptRules = PROMPT, max_len: int = SFT.max_len) -> tuple[list[Encoded], dict]:
    out, dropped, lengths = [], 0, []
    for e in examples:
        if not e.gold_ok:
            continue
        sch = schema_text(str(db_path(spider_dir, e.db_id, e.split)), rules.sample_rows, rules.max_cell_chars, rules.max_schema_chars)
        enc = encode(tok, messages(sch, e.question), target(e.query))
        lengths.append(len(enc.input_ids))
        if len(enc.input_ids) > max_len:
            dropped += 1
            continue
        out.append(enc)
    lens = np.asarray(lengths) if lengths else np.zeros(1)
    stats = {"kept": len(out), "dropped_too_long": dropped, "tokens_p50": float(np.percentile(lens, 50)), "tokens_p95": float(np.percentile(lens, 95)),
             "tokens_max": int(lens.max()), "target_tokens_total": int(sum(sum(1 for x in e.labels if x != IGNORE) for e in out))}
    return out, stats


def collate(batch: list[Encoded], pad_id: int):
    import torch

    width = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        mask[i, :n] = 1
    return ids, mask, labels


def add_lora(model, rules: LoraRules = LORA):
    from peft import LoraConfig, get_peft_model

    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in rules.target_modules if t in names]
    cfg = LoraConfig(r=rules.r, lora_alpha=rules.alpha, lora_dropout=rules.dropout, target_modules=targets, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def batches(data: list[Encoded], batch_size: int, rng: np.random.Generator):
    """Length-bucketed shuffled batches: sort within chunks of 50 batches so padding stays small."""
    order = rng.permutation(len(data))
    chunk = batch_size * 50
    out = []
    for s in range(0, len(order), chunk):
        part = sorted(order[s : s + chunk], key=lambda i: len(data[i].input_ids))
        out.extend([part[j : j + batch_size] for j in range(0, len(part), batch_size)])
    rng.shuffle(out)
    return out


def loss_on(model, batch, device, amp=None):
    """Summed cross-entropy over the target tokens, and their number. `logits_to_keep` gets the positions (1D, the
    union over the batch) that predict a target token, so the vocabulary projection runs only there — a 1D index is
    what transformers 4.5x and 5.x both accept. Rows with no target at a kept position are ignored. `amp`: autocast dtype."""
    import torch

    ids, mask, labels = (t.to(device) for t in batch)
    keep = torch.zeros_like(labels, dtype=torch.bool)
    keep[:, :-1] = labels[:, 1:] != IGNORE  # position t predicts token t + 1
    pos = keep.any(0).nonzero(as_tuple=True)[0]
    with torch.autocast(device_type="cuda", dtype=amp or torch.float16, enabled=amp is not None):
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=pos).logits.float()  # (batch, kept positions, vocab)
    tgt = labels[:, pos + 1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=IGNORE, reduction="sum")
    return loss, int((tgt != IGNORE).sum())


def evaluate_loss(model, data: list[Encoded], pad_id: int, device, batch_size: int, amp=None) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(data), batch_size):
            loss, n = loss_on(model, collate(data[s : s + batch_size], pad_id), device, amp)
            total, count = total + float(loss), count + n
    model.train()
    return total / max(count, 1)


def memory_probe(model, tok, data: list[Encoded], device, steps: int = 3, batch_size: int = 1, gradient_checkpointing: bool = True) -> dict:
    """Peak GPU memory of `steps` optimiser steps on the longest examples of `data` at `batch_size` — the same loss,
    precision and checkpointing as training. The optimiser is a throwaway with lr 0, so the weights do not move.
    Off CUDA there is nothing to measure (peaks are None)."""
    import torch

    from .generate import amp_dtype

    longest = sorted(data, key=lambda e: -len(e.input_ids))[: steps * batch_size]
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    amp = amp_dtype(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    cuda = device == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for s in range(0, len(longest), batch_size):
        loss, n = loss_on(model, collate(longest[s : s + batch_size], tok.pad_token_id), device, amp)
        scaler.scale(loss / max(n, 1)).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    out = {"batch_size": batch_size, "steps": len(range(0, len(longest), batch_size)), "tokens": [len(e.input_ids) for e in longest],
           "seconds": round(time.time() - t0, 1), "peak_allocated_gib": None, "peak_reserved_gib": None}
    if cuda:
        out["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    del opt
    if cuda:
        torch.cuda.empty_cache()
    return out


def choose_budget(lengths, probe, budgets=(2048, 3072, 4096), keep: float = 0.98, limit_gib: float = 14.0) -> tuple[int, dict]:
    """The smallest budget that keeps at least `keep` of the examples, if its probe (peak GPU memory in GiB, None when
    not measurable) stays under `limit_gib`; otherwise the largest smaller budget that fits."""
    lengths = np.asarray(lengths)
    ordered = sorted(budgets)
    share = {b: round(float((lengths <= b).mean()), 4) for b in ordered}
    first = next((b for b in ordered if share[b] >= keep), ordered[-1])
    report = {"kept_share": share, "keep_target": keep, "limit_gib": limit_gib, "probe_gib": {}}
    for b in [first] + [x for x in reversed(ordered) if x < first]:
        peak = probe(b)
        report["probe_gib"][b] = peak
        if peak is None or peak < limit_gib:
            report["chosen"] = b
            return b, report
    report["chosen"], report["note"] = ordered[0], "no budget stayed under the limit; using the smallest"
    return ordered[0], report


def train_sft(model, tok, train: list[Encoded], val: list[Encoded], out_dir: Path, device: str, rules: SFTRules = SFT, seed: int = SEED,
              gradient_checkpointing: bool = True, log=print, max_steps: int | None = None, max_hours: float | None = None, eta_after: int = 20,
              log_every: int = 20) -> dict:
    """Train the (already LoRA-wrapped) model in place; saves the best adapter to out_dir. Returns the training log.
    After `eta_after` optimiser steps it reports an ETA (training plus the val-loss checks, timed on the step-0 check);
    if that exceeds `max_hours` it drops to one epoch and records the switch. `log` rows carry the val loss (every
    `eval_every` steps, and at step 0), `train_log` rows the train loss every `log_every` steps."""
    import torch

    from .generate import amp_dtype

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    amp = amp_dtype(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rules.lr, weight_decay=rules.weight_decay)
    steps_per_epoch = math.ceil(len(train) / (rules.batch_size * rules.grad_accum))
    plan = {"total": max_steps or steps_per_epoch * rules.epochs, "epochs": rules.epochs}  # mutable: the ETA check may shorten it

    def lr_factor(s: int) -> float:
        warm = max(1, int(rules.warmup_ratio * plan["total"]))
        return min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(plan["total"], 1))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    pad = tok.pad_token_id
    t_eval = time.time()
    v0 = evaluate_loss(model, val, pad, device, rules.batch_size, amp) if val else float("nan")
    eval_s = time.time() - t_eval
    log_rows, train_rows, best, step, t0 = [{"step": 0, "epoch": 0.0, "train_loss": None, "val_loss": v0, "lr": 0.0, "minutes": 0.0}], [], float("inf"), 0, time.time()
    eta: dict = {}
    if log:
        log(f"step 0  val {v0:.4f}  (one val-loss check: {eval_s:.0f} s)  precision {str(amp).replace('torch.', '') if amp else 'fp32'}"
            f"{' + GradScaler' if scaler.is_enabled() else ''}")
    done = False
    for epoch in range(rules.epochs):
        bs = batches(train, rules.batch_size, rng)
        acc_loss, acc_n, win_loss, win_n = 0.0, 0, 0.0, 0
        for i, idx in enumerate(bs):
            loss, n = loss_on(model, collate([train[j] for j in idx], pad), device, amp)
            scaler.scale(loss / max(n, 1) / rules.grad_accum).backward()
            acc_loss, acc_n, win_loss, win_n = acc_loss + float(loss.detach()), acc_n + n, win_loss + float(loss.detach()), win_n + n
            if (i + 1) % rules.grad_accum == 0 or i == len(bs) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step == eta_after and not max_steps:
                    per_step = (time.time() - t0) / step
                    hours = (per_step * plan["total"] + eval_s * (plan["total"] // rules.eval_every + 1)) / 3600
                    eta = {"after_steps": step, "seconds_per_step": round(per_step, 2), "eta_hours": round(hours, 2), "planned_steps": plan["total"]}
                    if log:
                        log(f"ETA after {step} optimiser steps: {hours:.2f} h for {plan['total']} steps ({per_step:.1f} s/step)")
                    if max_hours and hours > max_hours and plan["epochs"] > 1:
                        plan.update(total=steps_per_epoch, epochs=1)
                        eta["switched_to_one_epoch"] = True
                        if log:
                            log(f"ETA {hours:.2f} h is over {max_hours} h: switching to 1 epoch ({plan['total']} steps)")
                if step % log_every == 0:
                    train_rows.append({"step": step, "train_loss": win_loss / max(win_n, 1), "lr": sched.get_last_lr()[0], "minutes": round((time.time() - t0) / 60, 1)})
                    win_loss, win_n = 0.0, 0
                if step % rules.eval_every == 0 or step == plan["total"]:
                    v = evaluate_loss(model, val, pad, device, rules.batch_size, amp) if val else float("nan")
                    row = {"step": step, "epoch": epoch + (i + 1) / len(bs), "train_loss": acc_loss / max(acc_n, 1), "val_loss": v, "lr": sched.get_last_lr()[0],
                           "minutes": round((time.time() - t0) / 60, 1)}
                    log_rows.append(row)
                    if log:
                        log(f"step {step}/{plan['total']}  train {row['train_loss']:.4f}  val {v:.4f}  {row['minutes']} min")
                    acc_loss, acc_n = 0.0, 0
                    if v < best or not val:
                        best = v
                        model.save_pretrained(out_dir)
                        tok.save_pretrained(out_dir)
                if step >= plan["total"]:
                    done = True
                    break
        if done:
            break
    info = {"steps": step, "epochs_run": plan["epochs"], "best_val_loss": best, "log": log_rows, "train_log": train_rows, "eta": eta,
            "precision": str(amp).replace("torch.", "") if amp else "fp32", "grad_scaler": scaler.is_enabled(), "minutes": round((time.time() - t0) / 60, 1),
            "rules": asdict(rules)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "sft_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
