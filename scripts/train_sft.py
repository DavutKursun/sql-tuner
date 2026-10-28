"""LoRA SFT on our train split; validation loss on val picks the saved adapter. Runs on Kaggle T4 (cuda, fp16) or,
for smoke runs, on CPU/MPS with the tiny model.

    python scripts/train_sft.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --out models/sft [--push-to DavutKursun/sql-tuner-sft]
    python scripts/train_sft.py --base tiny --max-steps 5 --out /tmp/sft     # offline smoke run
"""

from __future__ import annotations

import argparse
import json
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
from sqltuner.sft import add_lora, build_examples, train_sft


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
    ap.add_argument("--max-len", type=int, default=SFT.max_len)
    ap.add_argument("--eval-every", type=int, default=SFT.eval_every)
    ap.add_argument("--max-steps", type=int, default=None)
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
    train, s_tr = build_examples(train_ex, tok, sd, rules_p, a.max_len)
    val, s_va = build_examples(val_ex, tok, sd, rules_p, a.max_len)
    print(f"base {a.base} on {device}: trainable {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")
    print("train", s_tr, "\nval", s_va)
    rules = replace(SFT, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, grad_accum=a.grad_accum, max_len=a.max_len, eval_every=a.eval_every)
    info = train_sft(model, tok, train, val, a.out, device, rules, gradient_checkpointing=not a.no_gc, max_steps=a.max_steps)
    info.update({"base": a.base, "device": device, "trainable_params": trainable, "total_params": total, "train_stats": s_tr, "val_stats": s_va, "sample_rows": a.sample_rows,
                 "lora_r": a.lora_r})
    (a.out / "sft_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"train_sft_{a.out.name}.json").write_text(json.dumps({k: v for k, v in info.items() if k != "log"} | {"log_tail": info["log"][-5:]}, indent=2), encoding="utf-8")
    print(f"saved adapter to {a.out}; best val loss {info['best_val_loss']:.4f} after {info['steps']} steps, {info['minutes']} min")
    if a.push_to:
        from sqltuner.hub import push_folder

        print("pushed:", push_folder(a.out, a.push_to, message=f"SFT adapter, best val loss {info['best_val_loss']:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
