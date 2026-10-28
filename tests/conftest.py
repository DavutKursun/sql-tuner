from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import SplitRules
from sqltuner.data import build_splits, save_splits
from sqltuner.fixtures import make_spider


@pytest.fixture(scope="session")
def spider(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    sd = make_spider(raw)
    splits_dir = tmp_path_factory.mktemp("splits")
    splits = build_splits(sd, SplitRules(val_min_questions=20, val_max_dbs=3), log=None)
    summary = save_splits(splits, splits_dir)
    return {"raw": raw, "sd": sd, "splits": splits, "splits_dir": splits_dir, "summary": summary}


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    from sqltuner.tiny import build_tiny

    return build_tiny(tmp_path_factory.mktemp("tiny") / "m")
