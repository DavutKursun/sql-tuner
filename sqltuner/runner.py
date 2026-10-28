"""Evaluate a backend on a list of examples: build prompts, generate (greedy or sample-and-vote), execute, grade."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import GEN, PROMPT, SEED, PromptRules
from .data import Example, db_path
from .db import ExecResult, Executor
from .generate import Backend, Usage
from .metrics import accuracy, by_hardness, grade, majority_vote
from .prompts import extract_sql, messages, schema_text


@dataclass
class Prediction:
    id: str
    db_id: str
    hardness: str
    question: str
    gold: str
    pred: str
    correct: bool
    executed: bool
    error: str | None
    empty: bool
    n_samples: int = 1
    gold_empty: bool = False  # the gold result has no rows: any query that runs and returns none (same width) matches


def few_shot_pool(train: list[Example], spider_dir: Path, k: int, rules: PromptRules = PROMPT, seed: int = SEED) -> list[tuple[str, str, str, str]]:
    """k (db_id, schema, question, sql) examples from distinct training databases with short schemas and real rows."""
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    by_db: dict[str, list[Example]] = {}
    for e in train:
        if e.gold_ok and not e.gold_empty and not e.db_empty and e.hardness in ("medium", "hard"):
            by_db.setdefault(e.db_id, []).append(e)
    dbs = sorted(by_db)
    rng.shuffle(dbs)
    out = []
    for db in dbs:
        sch = schema_text(str(db_path(spider_dir, db, "train")), rules.sample_rows, rules.max_cell_chars, rules.max_schema_chars)
        if len(sch) > 2500:
            continue
        e = by_db[db][int(rng.integers(len(by_db[db])))]
        out.append((db, sch, e.question, e.query))
        if len(out) == k:
            break
    return out


def evaluate(examples: list[Example], backend: Backend, spider_dir: Path, rules: PromptRules = PROMPT, shots=(), samples: int = 1,
             temperature: float | None = None, max_new_tokens: int = GEN.max_new_tokens, chunk: int = 16, log=print) -> tuple[list[Prediction], Usage]:
    """samples = 1 → greedy (temperature 0 unless given); samples > 1 → sample at GEN.vote_temperature and majority-vote
    on execution results (self-consistency)."""
    ex = Executor()
    usage = Usage()
    preds: list[Prediction] = []
    temp = (0.0 if samples == 1 else GEN.vote_temperature) if temperature is None else temperature
    for start in range(0, len(examples), chunk):
        part = examples[start : start + chunk]
        convs = []
        for e in part:
            path = str(db_path(spider_dir, e.db_id, e.split))
            sch = schema_text(path, rules.sample_rows, rules.max_cell_chars, rules.max_schema_chars)
            convs.append(messages(sch, e.question, [(s, q, sql) for db, s, q, sql in shots if db != e.db_id]))
        replies = backend.generate(convs, n=samples, temperature=temp, max_new_tokens=max_new_tokens)
        for e, group in zip(part, replies):
            path = db_path(spider_dir, e.db_id, e.split)
            gold = ex.gold(path, e.query)
            sqls = [extract_sql(r.text) for r in group]
            results = [ex.run(path, s) if s else None for s in sqls]
            results = [r if r is not None else ExecResult(False, error="no sql") for r in results]
            pick = majority_vote(results) if samples > 1 else 0
            g = grade(results[pick], gold, e.query)
            for r in group:
                usage.add(r)
            preds.append(Prediction(e.id, e.db_id, e.hardness, e.question, e.query, sqls[pick], g.correct, g.executed, g.error, g.empty, samples, gold.empty))
        if log:
            done = len(preds)
            log(f"  {done:,}/{len(examples):,}  EX so far {np.mean([p.correct for p in preds]):.3f}")
    ex.close()
    return preds, usage


def summarize(preds: list[Prediction], usage: Usage | None = None, extra: dict | None = None) -> dict:
    """`accuracy` is EX on every question (Spider's convention); `ex_nonempty_gold` leaves out the questions whose gold
    result is empty, where a query that runs and returns nothing counts as correct."""
    correct = [p.correct for p in preds]
    nonempty = [p for p in preds if not p.gold_empty]
    errors: dict[str, int] = {}
    for p in preds:
        if p.error:
            key = p.error.split(":")[0] if not p.error.startswith("sqlite") else "sqlite: " + p.error.split(":", 1)[1].strip().split(" ")[0]
            errors[key] = errors.get(key, 0) + 1
    out = {"accuracy": accuracy(correct, [p.db_id for p in preds]),
           "ex_nonempty_gold": accuracy([p.correct for p in nonempty], [p.db_id for p in nonempty]) if nonempty else None,
           "by_hardness": by_hardness(correct, [p.hardness for p in preds]),
           "executed_rate": float(np.mean([p.executed for p in preds])) if preds else 0.0, "no_sql_rate": float(np.mean([not p.pred for p in preds])) if preds else 0.0,
           "errors": dict(sorted(errors.items(), key=lambda kv: -kv[1])[:10])}
    per_db = {}
    for p in preds:
        per_db.setdefault(p.db_id, []).append(p.correct)
    out["by_database"] = {db: {"n": len(v), "ex": float(np.mean(v))} for db, v in sorted(per_db.items())}
    if usage is not None:
        out["usage"] = usage.as_dict()
    if extra:
        out.update(extra)
    return out


def write_predictions(preds: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def read_predictions(path: Path) -> list[Prediction]:
    return [Prediction(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
