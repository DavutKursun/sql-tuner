"""LoRA SFT on our train split; validation loss on val picks the saved adapter. Runs in a Kaggle kernel (T4: fp16 +
GradScaler; bf16 on A100 / L4 / H100) or, for smoke runs, on CPU with the tiny model.

    python scripts/train_sft.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --max-len auto --batch-size 2 --grad-accum 8 --out models/sft
    python scripts/train_sft.py --base tiny --max-steps 5 --out /tmp/sft     # offline smoke run

--max-len auto: the smallest of 2,048 / 3,072 / 4,096 tokens that keeps >= 98% of train, if a memory probe (3 optimiser
steps on the longest kept examples, batch 1, gradient checkpointing) peaks under 14 GiB on the GPU; otherwise the largest
budget that fits. A second probe at the training batch size halves the batch (and doubles accumulation) if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import LORA, MODELS, PROMPT, RAW, RESULTS, SEED, SFT, SPLITS
from sqltuner.data import find_spider, load_split
from sqltuner.generate import load_model
from sqltuner.sft import IGNORE, add_lora, build_examples, choose_budget, memory_probe, train_sft

BUDGETS = (2048, 3072, 4096)
LIMIT_GIB = 14.0


def resolve_base(base: str) -> str:
    if base == "tiny":
        from sqltuner.tiny import build_tiny

        return str(build_tiny(Path(tempfile.mkdtemp()) / "tiny"))
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Hub id, local path, or 'tiny'")
    ap.add_argument("--out", type=Path, default=MODELS / "sft")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=SFT.epochs)
    ap.add_argument("--lr", type=float, default=SFT.lr)
    ap.add_argument("--batch-size", type=int, default=SFT.batch_size)
    ap.add_argument("--grad-accum", type=int, default=SFT.grad_accum)
    ap.add_argument("--max-len", default=str(SFT.max_len), help="tokens per example, or 'auto' (see above)")
    ap.add_argument("--eval-every", type=int, default=SFT.eval_every)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None, help="after 20 optimiser steps, drop to 1 epoch if the ETA is longer")
    ap.add_argument("--lora-r", type=int, default=LORA.r)
    ap.add_argument("--sample-rows", type=int, default=PROMPT.sample_rows)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=300, help="val questions used for the validation loss")
    ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--push-to", default=None, help="Hub model repo for the adapter (token from env / hf login, never printed)")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    sd = find_spider(a.raw)
    if sd is None:
        print("Spider not found; run scripts/download.py", file=sys.stderr)
        return 1
    base = resolve_base(a.base)
    tok, model, device = load_model(base, device=a.device, trainable=True)
    tok.padding_side = "right"
    model = add_lora(model, replace(LORA, r=a.lora_r, alpha=2 * a.lora_r))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rules_p = replace(PROMPT, sample_rows=a.sample_rows)
    train_ex = load_split("train", a.splits_dir)
    if a.train_limit:
        train_ex = [train_ex[i] for i in np.random.default_rng(SEED).choice(len(train_ex), min(a.train_limit, len(train_ex)), replace=False)]
    val_ex = load_split("val", a.splits_dir)
    val_ex = [val_ex[i] for i in np.random.default_rng(SEED).choice(len(val_ex), min(a.val_limit, len(val_ex)), replace=False)]
    print(f"base {a.base} on {device}: trainable {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")
    budget_report, batch_size, grad_accum = None, a.batch_size, a.grad_accum
    if a.max_len == "auto":
        every, _ = build_examples(train_ex, tok, sd, rules_p, max_len=10**9)
        lengths = [len(e.input_ids) for e in every]

        def probe(b: int):
            r = memory_probe(model, tok, [e for e in every if len(e.input_ids) <= b], device, steps=3, batch_size=1, gradient_checkpointing=not a.no_gc)
            print(f"memory probe at max_len {b}: 3 steps, batch 1, tokens {r['tokens']}: peak allocated {r['peak_allocated_gib']} GiB, "
                  f"reserved {r['peak_reserved_gib']} GiB ({r['seconds']} s)")
            return r["peak_reserved_gib"]

        max_len, budget_report = choose_budget(lengths, probe, BUDGETS, keep=0.98, limit_gib=LIMIT_GIB)
        print("length budget:", json.dumps(budget_report))
        if batch_size > 1:  # the training batch on the longest kept examples must fit too; else halve it, keeping sequences per step
            r = memory_probe(model, tok, [e for e in every if len(e.input_ids) <= max_len], device, steps=2, batch_size=batch_size, gradient_checkpointing=not a.no_gc)
            budget_report["probe_training_batch"] = r
            print(f"probe at the training batch ({batch_size}): reserved {r['peak_reserved_gib']} GiB")
            while r["peak_reserved_gib"] is not None and r["peak_reserved_gib"] >= LIMIT_GIB and batch_size > 1:
                batch_size, grad_accum = batch_size // 2, grad_accum * 2
                r = memory_probe(model, tok, [e for e in every if len(e.input_ids) <= max_len], device, steps=2, batch_size=batch_size, gradient_checkpointing=not a.no_gc)
                print(f"  -> batch {batch_size} x accumulation {grad_accum}: reserved {r['peak_reserved_gib']} GiB")
            budget_report["batch_size"], budget_report["grad_accum"] = batch_size, grad_accum
        train = [e for e in every if len(e.input_ids) <= max_len]
        lens = np.asarray([len(e.input_ids) for e in train])
        s_tr = {"kept": len(train), "dropped_too_long": len(every) - len(train), "tokens_p50": float(np.percentile(lens, 50)),
                "tokens_p95": float(np.percentile(lens, 95)), "tokens_max": int(lens.max()),
                "target_tokens_total": int(sum(sum(1 for x in e.labels if x != IGNORE) for e in train))}
    else:
        max_len = int(a.max_len)
        train, s_tr = build_examples(train_ex, tok, sd, rules_p, max_len)
    val, s_va = build_examples(val_ex, tok, sd, rules_p, max_len)
    print("train", s_tr, "\nval", s_va)
    rules = replace(SFT, epochs=a.epochs, lr=a.lr, batch_size=batch_size, grad_accum=grad_accum, max_len=max_len, eval_every=a.eval_every)
    info = train_sft(model, tok, train, val, a.out, device, rules, gradient_checkpointing=not a.no_gc, max_steps=a.max_steps, max_hours=a.max_hours)
    info.update({"base": a.base, "device": device, "trainable_params": trainable, "total_params": total, "train_stats": s_tr, "val_stats": s_va, "sample_rows": a.sample_rows,
                 "lora_r": a.lora_r, "length_budget": budget_report})
    steps_per_epoch = max(math.ceil(len(train) / (batch_size * grad_accum)), 1)
    info["tokens_seen"] = int(sum(len(e.input_ids) for e in train) * info["steps"] / steps_per_epoch)
    info["tokens_per_second"] = round(info["tokens_seen"] / max(60 * info["minutes"], 1e-9), 1)  # wall clock, val checks included
    (a.out / "sft_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"train_sft_{a.out.name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")  # the whole curve: a few dozen rows
    print(f"saved adapter to {a.out}; best val loss {info['best_val_loss']:.4f} after {info['steps']} steps, {info['minutes']} min")
    if a.push_to:
        from sqltuner.hub import push_folder

        print("pushed:", push_folder(a.out, a.push_to, message=f"SFT adapter, best val loss {info['best_val_loss']:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
