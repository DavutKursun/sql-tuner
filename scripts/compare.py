"""Paired comparison of two prediction files on the same questions: difference in EX with CIs by question and by
database, plus where each one wins (hardness).

    python scripts/compare.py eval/results/val/A.jsonl eval/results/val/B.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.metrics import paired
from sqltuner.runner import read_predictions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    pa = {p.id: p for p in read_predictions(a.a)}
    pb = {p.id: p for p in read_predictions(a.b)}
    ids = sorted(set(pa) & set(pb))
    if not ids:
        print("no common questions", file=sys.stderr)
        return 1
    res = paired([pa[i].correct for i in ids], [pb[i].correct for i in ids], [pa[i].db_id for i in ids])
    by = {}
    for i in ids:
        h = pa[i].hardness
        d = by.setdefault(h, {"n": 0, "a": 0, "b": 0})
        d["n"] += 1
        d["a"] += pa[i].correct
        d["b"] += pb[i].correct
    res.update({"a": a.a.stem, "b": a.b.stem, "n": len(ids), "by_hardness": {h: {"n": v["n"], "a": v["a"] / v["n"], "b": v["b"] / v["n"]} for h, v in by.items()}})
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
