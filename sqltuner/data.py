"""Spider files → our splits. Every split is a list of examples {id, split, db_id, question, query, hardness, gold_ok,
gold_rows, gold_empty, gold_truncated, db_empty} written as JSONL under data/splits/.

  train = Spider train_spider + train_others, minus the held-out validation databases
  val   = whole Spider training databases held out (cross-database, like dev): every choice is made here
  dev   = Spider dev (20 databases never seen in training): the reported number, read once in M4
  test  = Spider test, only if the release contains it: read once in M4
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import EXEC, RAW, SEED, SPIDER_DIRNAME, SPLIT_RULES, SPLITS, ExecRules, SplitRules
from .db import Executor, db_is_empty
from .hardness import hardness

SPLIT_NAMES = ("train", "val", "dev", "test")
CLOSED_SPLITS = ("dev", "test")  # need --final


@dataclass
class Example:
    id: str
    split: str
    db_id: str
    question: str
    query: str
    hardness: str = "unknown"
    gold_ok: bool = True
    gold_rows: int = 0
    gold_empty: bool = False
    gold_truncated: bool = False  # the gold result has more rows than ExecRules.max_rows
    db_empty: bool = False  # every table of the database has 0 rows (kept for SFT, left out of the GRPO prompt pool)

    def to_dict(self) -> dict:
        return asdict(self)


# --- locating the raw files -----------------------------------------------------------------------------------------
def find_spider(raw_dir: Path = RAW) -> Path | None:
    """The folder that holds train_spider.json and database/ (directly in data/raw or one level below)."""
    for cand in [raw_dir / SPIDER_DIRNAME, raw_dir, *[p for p in raw_dir.glob("*/") if p.is_dir()], *[p for p in raw_dir.glob("*/*/") if p.is_dir()]]:
        if (cand / "train_spider.json").exists() and (cand / "database").is_dir():
            return cand
    return None


def db_path(spider_dir: Path, db_id: str, split: str) -> Path:
    """Test questions use test_database/ when the release has it; everything else database/."""
    if split == "test" and (spider_dir / "test_database" / db_id / f"{db_id}.sqlite").exists():
        return spider_dir / "test_database" / db_id / f"{db_id}.sqlite"
    return spider_dir / "database" / db_id / f"{db_id}.sqlite"


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def raw_files(spider_dir: Path) -> dict[str, list[dict]]:
    return {"train_spider": _read(spider_dir / "train_spider.json"), "train_others": _read(spider_dir / "train_others.json"),
            "dev": _read(spider_dir / "dev.json"), "test": _read(spider_dir / "test.json")}


# --- splits ---------------------------------------------------------------------------------------------------------
def choose_val_dbs(train: list[dict], rules: SplitRules = SPLIT_RULES, seed: int = SEED) -> list[str]:
    """Whole databases, picked at random until ≥ val_min_questions (capped at val_max_dbs). Only train_spider databases
    are eligible, so the val distribution looks like dev."""
    counts: dict[str, int] = {}
    for ex in train:
        counts[ex["db_id"]] = counts.get(ex["db_id"], 0) + 1
    dbs = sorted(counts)
    order = np.random.default_rng(seed).permutation(len(dbs))
    chosen, total = [], 0
    for i in order:
        db = dbs[i]
        if counts[db] > 3 * rules.val_min_questions / max(rules.val_max_dbs, 1):  # skip giants so val stays varied
            continue
        chosen.append(db)
        total += counts[db]
        if total >= rules.val_min_questions or len(chosen) >= rules.val_max_dbs:
            break
    return sorted(chosen)


def build_splits(spider_dir: Path, rules: SplitRules = SPLIT_RULES, execute_gold: bool = True, log=print,
                 exec_rules: ExecRules = EXEC) -> dict[str, list[Example]]:
    raw = raw_files(spider_dir)
    val_dbs = set(choose_val_dbs(raw["train_spider"], rules))
    out: dict[str, list[Example]] = {s: [] for s in SPLIT_NAMES}
    sources = [("train_spider", "train"), ("train_others", "train"), ("dev", "dev"), ("test", "test")]
    for src, split in sources:
        for i, ex in enumerate(raw[src]):
            s = "val" if split == "train" and ex["db_id"] in val_dbs else split
            out[s].append(Example(id=f"{src}-{i}", split=s, db_id=ex["db_id"], question=ex["question"].strip(), query=ex["query"].strip()))
    ex_ = Executor(exec_rules)
    empty: dict[Path, bool] = {}
    for s, items in out.items():
        for e in items:
            e.hardness = hardness(e.query)
            path = db_path(spider_dir, e.db_id, s)
            if path not in empty:
                empty[path] = db_is_empty(path)
            e.db_empty = empty[path]
            if execute_gold:
                r = ex_.gold(path, e.query)
                e.gold_ok, e.gold_rows, e.gold_empty, e.gold_truncated = r.ok, len(r.rows), r.empty, r.truncated
        if log and items:
            log(f"{s:6s} {len(items):6,} questions on {len({e.db_id for e in items}):3d} databases; gold failures {sum(not e.gold_ok for e in items)}; "
                f"on empty databases {sum(e.db_empty for e in items)}")
    ex_.close()
    return out


def save_splits(splits: dict[str, list[Example]], out_dir: Path = SPLITS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"splits": {}}
    for s, items in splits.items():
        with (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") as f:
            for e in items:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        hard = {}
        for e in items:
            hard[e.hardness] = hard.get(e.hardness, 0) + 1
        summary["splits"][s] = {"questions": len(items), "databases": len({e.db_id for e in items}), "hardness": hard,
                                "gold_failures": sum(not e.gold_ok for e in items), "gold_empty": sum(e.gold_empty for e in items),
                                "gold_truncated": sum(e.gold_truncated for e in items), "db_empty": sum(e.db_empty for e in items),
                                "gold_empty_on_nonempty_dbs": sum(e.gold_empty and not e.db_empty for e in items)}
    summary["empty_dbs"] = sorted({e.db_id for items in splits.values() for e in items if e.db_empty})
    summary["val_dbs"] = sorted({e.db_id for e in splits.get("val", [])})
    overlap = {e.db_id for e in splits.get("train", [])} & {e.db_id for e in splits.get("val", []) + splits.get("dev", [])}
    summary["db_overlap_train_vs_val_dev"] = sorted(overlap)
    train_q = {e.question.lower() for e in splits.get("train", [])}
    summary["duplicate_questions_in_train"] = {s: sum(e.question.lower() in train_q for e in splits.get(s, [])) for s in ("val", "dev", "test")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def stratified_subset(examples: list[Example], n: int, seed: int = SEED) -> list[Example]:
    """n examples, each database contributing in proportion to its size (largest remainder), drawn with `seed`; returned
    in their original order. GRPO tracks greedy EX on such a fixed val subset during training."""
    by_db: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        by_db.setdefault(e.db_id, []).append(i)
    n = min(n, len(examples))
    quota = {db: n * len(ix) / len(examples) for db, ix in by_db.items()}
    take = {db: int(q) for db, q in quota.items()}
    for db in sorted(quota, key=lambda d: (take[d] - quota[d], d))[: n - sum(take.values())]:
        take[db] += 1
    rng = np.random.default_rng(seed)
    chosen = [int(i) for db in sorted(by_db) for i in rng.choice(by_db[db], size=take[db], replace=False)]
    return [examples[i] for i in sorted(chosen)]


def load_split(name: str, splits_dir: Path = SPLITS) -> list[Example]:
    path = splits_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_splits.py")
    return [Example(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
