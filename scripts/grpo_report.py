"""Tables for DESIGN.md §4 from committed GRPO results (no GPU): training monitors by window of steps, execution
failures, hacking-flag shares with real examples, val-subset EX by step and the checkpoint choice (ties: the earlier
step), the ablation at its last common step, and which of M2's 30 categorised SFT failures the GRPO model fixes.

    python scripts/grpo_report.py --main eval/results/train_grpo_grpo-main.json [--ablation eval/results/train_grpo_grpo-ablation.json] \
        [--grpo-preds eval/results/val/<chosen>.jsonl] --out eval/results/grpo_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.metrics import FAILURE_KINDS
from sqltuner.runner import read_predictions

FLAGS = ("empty_result", "added_limit", "select_star", "row_cap", "row_flood", "truncated")
MONITORS = ("reward", "correct", "executed", "executed_wrong", "zero_var_groups", "kl", "completion_tokens", "completion_tokens_p95", "truncated",
            "gen_s", "reward_s", "ref_s", "train_s", "peak_gib")


def windows(history: list[dict], size: int = 50) -> list[dict]:
    """Mean of every monitor, failure kind and flag over consecutive windows of `size` steps."""
    out = []
    for s in range(0, len(history), size):
        rows = history[s : s + size]
        w = {"steps": f"{rows[0]['step']}-{rows[-1]['step']}"}
        for m in MONITORS:
            vals = [r[m] for r in rows if r.get(m) is not None]
            w[m] = round(float(np.mean(vals)), 4) if vals else None
        for k in FAILURE_KINDS:
            w["fail_" + k] = round(float(np.mean([r.get("failures", {}).get(k, 0.0) for r in rows])), 4)
        for f in FLAGS:
            w["flag_" + f] = round(float(np.mean([r.get("flags", {}).get(f, 0.0) for r in rows])), 4)
        out.append(w)
    return out


def choose(evals: list[dict]) -> dict:
    """The checkpoint with the highest val-subset EX; a tie goes to the earlier step."""
    best = max(evals, key=lambda e: (e["val_ex"], -e["step"]))
    return {"step": best["step"], "val_ex": best["val_ex"]}


def fixed_failures(analysis: Path, preds: Path) -> dict:
    """Of the hand-categorised SFT failures, how many the GRPO predictions get right, by category."""
    cats = {i["id"]: i["category"] for i in json.loads(analysis.read_text())["items"]}
    got = {p.id: p.correct for p in read_predictions(preds)}
    by: dict[str, dict] = {}
    for pid, c in cats.items():
        d = by.setdefault(c, {"n": 0, "fixed": 0})
        d["n"] += 1
        d["fixed"] += int(got.get(pid, False))
    return {"total": len(cats), "fixed": sum(d["fixed"] for d in by.values()), "by_category": by, "fixed_ids": sorted(p for p in cats if got.get(p))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ablation", type=Path, default=None)
    ap.add_argument("--grpo-preds", type=Path, default=None, help="full-val predictions of the chosen checkpoint")
    ap.add_argument("--analysis", type=Path, default=ROOT / "eval" / "results" / "val" / "sft_error_analysis.json")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    main_log = json.loads(a.main.read_text())
    rep = {"steps": main_log["steps"], "stopped": main_log.get("stopped"), "minutes": main_log.get("minutes"), "evals": main_log["evals"],
           "chosen": choose(main_log["evals"]), "windows": windows(main_log["history"], a.window),
           "flag_examples": {f: main_log.get("examples", {}).get(f, [])[:2] for f in FLAGS}}
    if a.ablation:
        abl = json.loads(a.ablation.read_text())
        common = max(e["step"] for e in abl["evals"] if e["step"] in {m["step"] for m in main_log["evals"]})
        n = min(len(abl["history"]), common)

        def pick(h: list[dict], k: str) -> float:
            return round(float(np.mean([r.get(k, 0.0) for r in h[:n]])), 4)

        rep["ablation"] = {"common_step": common, "val_ex": {"main": next(e["val_ex"] for e in main_log["evals"] if e["step"] == common),
                                                             "ablation": next(e["val_ex"] for e in abl["evals"] if e["step"] == common)},
                           "monitors_steps_1_to_common": {k: {"main": pick(main_log["history"], k), "ablation": pick(abl["history"], k)}
                                                          for k in ("reward", "correct", "executed", "executed_wrong", "zero_var_groups", "kl", "completion_tokens")},
                           "failure_share_1_to_common": {"main": round(float(np.mean([sum(r.get("failures", {}).values()) for r in main_log["history"][:n]])), 4),
                                                         "ablation": round(float(np.mean([sum(r.get("failures", {}).values()) for r in abl["history"][:n]])), 4)}}
    if a.grpo_preds:
        rep["sft_failures_fixed"] = fixed_failures(a.analysis, a.grpo_preds)
    text = json.dumps(rep, indent=1)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
