"""Supervised fine-tuning with LoRA, written as a plain PyTorch loop (no Trainer), so every step is visible:

    prompt = chat template(system, schema + question) with the generation prompt
    target = "```sql\n<gold>\n```" + end-of-turn token
    loss   = cross-entropy on the target tokens only (prompt tokens get label −100)

fp16 autocast + GradScaler on CUDA (Kaggle T4 has no bf16), fp32 on CPU/MPS; gradient checkpointing; gradient
accumulation; cosine schedule with warmup; the adapter with the lowest validation loss is kept.
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


def loss_on(model, batch, device, use_amp: bool):
    import torch

    ids, mask, labels = (t.to(device) for t in batch)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[:, :-1].float()
        tgt = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=IGNORE, reduction="sum")
    n = (tgt != IGNORE).sum()
    return loss, n


def evaluate_loss(model, data: list[Encoded], pad_id: int, device, batch_size: int, use_amp: bool) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(data), batch_size):
            loss, n = loss_on(model, collate(data[s : s + batch_size], pad_id), device, use_amp)
            total, count = total + float(loss), count + int(n)
    model.train()
    return total / max(count, 1)


def train_sft(model, tok, train: list[Encoded], val: list[Encoded], out_dir: Path, device: str, rules: SFTRules = SFT, seed: int = SEED,
              gradient_checkpointing: bool = True, log=print, max_steps: int | None = None) -> dict:
    """Train the (already LoRA-wrapped) model in place; saves the best adapter to out_dir. Returns the training log."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_amp = device == "cuda"
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rules.lr, weight_decay=rules.weight_decay)
    steps_per_epoch = math.ceil(len(train) / (rules.batch_size * rules.grad_accum))
    total = max_steps or steps_per_epoch * rules.epochs
    warm = max(1, int(rules.warmup_ratio * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_rows, best, step, t0 = [], float("inf"), 0, time.time()
    pad = tok.pad_token_id
    done = False
    for epoch in range(rules.epochs):
        bs = batches(train, rules.batch_size, rng)
        acc_loss, acc_n = 0.0, 0
        for i, idx in enumerate(bs):
            loss, n = loss_on(model, collate([train[j] for j in idx], pad), device, use_amp)
            scaler.scale(loss / max(int(n), 1) / rules.grad_accum).backward()
            acc_loss, acc_n = acc_loss + float(loss.detach()), acc_n + int(n)
            if (i + 1) % rules.grad_accum == 0 or i == len(bs) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % rules.eval_every == 0 or step == total:
                    v = evaluate_loss(model, val, pad, device, rules.batch_size, use_amp) if val else float("nan")
                    row = {"step": step, "epoch": epoch + (i + 1) / len(bs), "train_loss": acc_loss / max(acc_n, 1), "val_loss": v, "lr": sched.get_last_lr()[0],
                           "minutes": round((time.time() - t0) / 60, 1)}
                    log_rows.append(row)
                    if log:
                        log(f"step {step}/{total}  train {row['train_loss']:.4f}  val {v:.4f}  {row['minutes']} min")
                    acc_loss, acc_n = 0.0, 0
                    if v < best or not val:
                        best = v
                        model.save_pretrained(out_dir)
                        tok.save_pretrained(out_dir)
                if step >= total:
                    done = True
                    break
        if done:
            break
    info = {"steps": step, "best_val_loss": best, "log": log_rows, "minutes": round((time.time() - t0) / 60, 1), "rules": asdict(rules)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "sft_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
