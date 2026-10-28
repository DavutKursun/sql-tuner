"""Execution accuracy of one model on one split, with question- and database-level CIs, hardness breakdown, error
types, latency and (for the API) cost. dev and test need --final and are each run once in Milestone 4.

    python scripts/eval_model.py --backend hf:Qwen/Qwen2.5-Coder-1.5B-Instruct --split val [--limit 300]
    python scripts/eval_model.py --backend hf:<base> --adapter models/sft --split val
    python scripts/eval_model.py --backend anthropic:claude-haiku-5-5 --split val --few-shot 3 --max-usd 2
    python scripts/eval_model.py --backend hf:<base> --adapter models/grpo --split val --samples 8   # self-consistency
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import GEN, PROMPT, RAW, RESULTS, SEED, SPLITS
from sqltuner.data import CLOSED_SPLITS, find_spider, load_split
from sqltuner.generate import GoldBackend, make_backend
from sqltuner.runner import evaluate, few_shot_pool, summarize, write_predictions


def tag_for(a) -> str:
    if a.tag:
        return a.tag
    base = a.backend.split(":", 1)[-1].split("/")[-1]
    parts = [base]
    if a.adapter:
        parts.append(Path(a.adapter.rstrip("/")).name)
    if a.few_shot:
        parts.append(f"{a.few_shot}shot")
    if a.samples > 1:
        parts.append(f"vote{a.samples}")
    if a.sample_rows != PROMPT.sample_rows:
        parts.append(f"rows{a.sample_rows}")
    if a.limit:
        parts.append(f"n{a.limit}")
    return "-".join(parts)


def main(argv=None, backend_override=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="hf:<id or path> | anthropic:<model> | llamacpp:<gguf> | gold")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "dev", "test"])
    ap.add_argument("--final", action="store_true", help="required for dev and test (read once, Milestone 4)")
    ap.add_argument("--limit", type=int, default=0, help="random subset of this many questions (seeded); 0 = all")
    ap.add_argument("--samples", type=int, default=1, help=">1 = sample and majority-vote on execution results")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--sample-rows", type=int, default=PROMPT.sample_rows)
    ap.add_argument("--max-new-tokens", type=int, default=GEN.max_new_tokens)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split in CLOSED_SPLITS and not a.final:
        print(f"refusing to read {a.split} without --final (it is read once, in Milestone 4)", file=sys.stderr)
        return 3
    sd = find_spider(a.raw)
    if sd is None:
        print("Spider not found; run scripts/download.py", file=sys.stderr)
        return 1
    examples = load_split(a.split, a.splits_dir)
    if a.limit and a.limit < len(examples):
        idx = np.sort(np.random.default_rng(SEED).choice(len(examples), a.limit, replace=False))
        examples = [examples[i] for i in idx]
    rules = replace(PROMPT, sample_rows=a.sample_rows)
    shots = few_shot_pool(load_split("train", a.splits_dir), sd, a.few_shot, rules) if a.few_shot else []
    if a.backend.startswith("anthropic:"):
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")  # the key stays in .env; nothing is printed
        except ImportError:
            pass
    if backend_override is not None:
        backend = backend_override
    elif a.backend == "gold":
        backend = GoldBackend([e.query for e in examples])
    else:
        backend = make_backend(a.backend, a.adapter, a.device, a.max_usd, a.batch_size)
    t = time.time()
    preds, usage = evaluate(examples, backend, sd, rules, shots, a.samples, max_new_tokens=a.max_new_tokens)
    tag = tag_for(a)
    out_dir = a.out / a.split
    write_predictions(preds, out_dir / f"{tag}.jsonl")
    summary = summarize(preds, usage, {"backend": a.backend, "adapter": a.adapter, "split": a.split, "tag": tag, "few_shot": a.few_shot, "samples": a.samples,
                                       "sample_rows": a.sample_rows, "limit": a.limit, "run_date": str(date.today()), "minutes": round((time.time() - t) / 60, 2)})
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    acc = summary["accuracy"]
    print(f"{tag} on {a.split}: EX {acc['ex']:.4f}  CI by question [{acc['ci_case'][0]:.4f}, {acc['ci_case'][1]:.4f}]  "
          f"CI by database [{acc['ci_db'][0]:.4f}, {acc['ci_db'][1]:.4f}]  n={acc['n']} dbs={acc['n_db']}")
    ne = summary["ex_nonempty_gold"]
    if ne:
        print(f"  non-empty gold: EX {ne['ex']:.4f}  CI by question [{ne['ci_case'][0]:.4f}, {ne['ci_case'][1]:.4f}]  "
              f"CI by database [{ne['ci_db'][0]:.4f}, {ne['ci_db'][1]:.4f}]  n={ne['n']}")
    print("by hardness:", {k: (v["n"], round(v["ex"], 3) if v["ex"] is not None else None) for k, v in summary["by_hardness"].items()})
    print("executed", round(summary["executed_rate"], 3), "| errors", summary["errors"], "| usage", summary.get("usage"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
