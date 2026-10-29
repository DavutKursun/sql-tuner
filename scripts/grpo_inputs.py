"""GRPO inputs that need no GPU: the prompt pool's size, what was dropped and why, its hardness mix (with the base
model's tokenizer only), and the fixed val subset GRPO tracks during training (stratified by database, seeded).

    python scripts/grpo_inputs.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct
    -> eval/results/grpo_pool.json, eval/results/val_subset_ids.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import GRPO, RAW, RESULTS, SEED, SPLITS
from sqltuner.data import find_spider, load_split, stratified_subset

sys.path.insert(0, str(ROOT / "scripts"))
from train_grpo import build_pool, hardness_mix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="its tokenizer measures prompt length")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--subset", type=int, default=GRPO.eval_questions)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    train = load_split("train", a.splits_dir)
    pool, _, skipped = build_pool(tok, find_spider(a.raw), a.splits_dir, GRPO, a.max_prompt_tokens)
    info = {"tokenizer": a.base, "max_prompt_tokens": a.max_prompt_tokens, "train": len(train), "pool": len(pool), "dropped": skipped,
            "train_hardness": hardness_mix(train), "pool_hardness": hardness_mix(pool)}
    val = stratified_subset(load_split("val", a.splits_dir), a.subset, a.seed)
    by_db = Counter(e.db_id for e in val)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "grpo_pool.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (a.out / "val_subset_ids.json").write_text(json.dumps({"seed": a.seed, "n": len(val), "stratified_by": "db_id", "by_db": dict(sorted(by_db.items())),
                                                          "hardness": hardness_mix(val), "ids": [e.id for e in val]}, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    print("val subset:", len(val), dict(sorted(by_db.items())), hardness_mix(val))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
