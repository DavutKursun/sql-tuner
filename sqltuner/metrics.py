"""Execution accuracy with two kinds of confidence interval, paired differences, breakdowns, and the RL reward.

Two intervals, because they answer different questions:
  * case bootstrap      — "how sure are we about accuracy on THESE databases?" (resample questions)
  * database bootstrap  — "how sure are we about accuracy on a NEW database?" (resample whole databases, Spider dev has
                          only 20, so this interval is wide; it is the honest one for a client with their own schema)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BOOTSTRAP_RESAMPLES, GRPO, HARDNESS_LEVELS, SEED
from .db import ExecResult, has_order_by, results_match


@dataclass
class Graded:
    correct: bool
    executed: bool  # the prediction ran without error
    error: str | None
    empty: bool  # the prediction returned no rows
    pred_rows: int


def grade(pred: ExecResult, gold: ExecResult, gold_sql: str) -> Graded:
    ok = results_match(pred, gold, ordered=has_order_by(gold_sql))
    return Graded(ok, pred.ok, pred.error, pred.empty, len(pred.rows))


def _ci(values: np.ndarray, clusters: np.ndarray | None, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(values), size=(n, len(values)))
        boot = values[idx].mean(axis=1)
    else:
        uniq, inv = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inv, weights=values, minlength=len(uniq))
        cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
        boot = np.empty(n)
        for b in range(n):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            boot[b] = (w @ sums) / max(w @ cnts, 1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def accuracy(correct, db_ids, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict:
    v = np.asarray(correct, dtype=float)
    lo, hi = _ci(v, None, n_resamples, seed)
    dlo, dhi = _ci(v, np.asarray(db_ids), n_resamples, seed)
    return {"ex": float(v.mean()), "ci_case": [lo, hi], "ci_db": [dlo, dhi], "n": len(v), "n_db": len(set(db_ids))}


def paired(a, b, db_ids, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict:
    """mean(a) − mean(b) on the same questions; intervals by question and by database."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = _ci(d, None, n_resamples, seed)
    dlo, dhi = _ci(d, np.asarray(db_ids), n_resamples, seed)
    return {"diff": float(d.mean()), "ci_case": [lo, hi], "ci_db": [dlo, dhi], "a_only": int(((d > 0)).sum()), "b_only": int((d < 0).sum()),
            "significant_case": bool(lo > 0 or hi < 0), "significant_db": bool(dlo > 0 or dhi < 0)}


def by_hardness(correct, levels) -> dict:
    c, lv = np.asarray(correct, dtype=float), np.asarray(levels)
    return {h: {"n": int((lv == h).sum()), "ex": float(c[lv == h].mean()) if (lv == h).any() else None} for h in (*HARDNESS_LEVELS, "unknown") if (lv == h).any()}


def majority_vote(results: list[ExecResult]) -> int:
    """Index of the sample whose result set is shared by most samples (errors never win; ties → earliest)."""
    keys = []
    for r in results:
        keys.append(None if not r.ok else (len(r.columns), tuple(sorted(map(repr, r.rows)))))
    best, best_n = 0, -1
    for i, k in enumerate(keys):
        if k is None:
            continue
        n = sum(1 for kk in keys if kk == k)
        if n > best_n:
            best, best_n = i, n
    return best


# --- the RL reward ----------------------------------------------------------------------------------------------------
@dataclass
class Reward:
    value: float
    correct: bool
    executed: bool
    flags: tuple[str, ...]  # reward-hacking monitors, never part of the value


def reward(pred_sql: str, pred: ExecResult, gold: ExecResult, gold_sql: str, exec_bonus: float = GRPO.exec_bonus) -> Reward:
    """1 for the right result; `exec_bonus` for SQL that runs but is wrong; 0 for no SQL, a refused or failing query.
    Flags record behaviour worth watching during training (they are logged, not rewarded or punished)."""
    if not pred_sql:
        return Reward(0.0, False, False, ("no_sql",))
    g = grade(pred, gold, gold_sql)
    flags = []
    if g.executed and g.empty and gold.ok and not gold.empty:
        flags.append("empty_result")
    low = pred_sql.lower()
    if " limit " in f" {low} " and " limit " not in f" {gold_sql.lower()} ":
        flags.append("added_limit")
    if "select *" in low and "select *" not in gold_sql.lower():
        flags.append("select_star")
    if g.executed and gold.ok and len(gold.rows) and g.pred_rows > 10 * len(gold.rows):
        flags.append("row_flood")
    if pred.error and pred.error.startswith("guard"):
        flags.append("guard_refused")
    if pred.error == "timeout":
        flags.append("timeout")
    value = 1.0 if g.correct else (exec_bonus if g.executed else 0.0)
    return Reward(value, g.correct, g.executed, tuple(flags))
