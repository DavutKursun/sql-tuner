"""Spider → data/splits/{train,val,dev,test}.jsonl + summary.json (gold queries executed once, hardness computed).

    python scripts/build_splits.py [--raw data/raw] [--val-min 600] [--no-exec]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import RAW, SPLIT_RULES, SPLITS, SplitRules
from sqltuner.data import build_splits, find_spider, save_splits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=SPLITS)
    ap.add_argument("--val-min", type=int, default=SPLIT_RULES.val_min_questions)
    ap.add_argument("--val-max-dbs", type=int, default=SPLIT_RULES.val_max_dbs)
    ap.add_argument("--no-exec", action="store_true", help="skip executing gold queries")
    a = ap.parse_args(argv)
    sd = find_spider(a.raw)
    if sd is None:
        print("Spider not found; run scripts/download.py", file=sys.stderr)
        return 1
    t = time.time()
    splits = build_splits(sd, SplitRules(a.val_min, a.val_max_dbs), execute_gold=not a.no_exec)
    summary = save_splits(splits, a.out)
    summary["seconds"] = round(time.time() - t, 1)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
