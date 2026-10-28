"""The script chain on the miniature Spider and the tiny model: download check → build_splits → eval (gold, closed splits
refused) → train_sft → eval with the adapter → train_grpo from the SFT adapter → compare → Kaggle notebook. Same chain
as on the real data, so the first real run is not the first run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_splits
import compare
import download
import eval_model
import make_kaggle_notebook
import run_hf_job
import train_grpo
import train_sft

from sqltuner.data import load_split
from sqltuner.fixtures import make_spider


@pytest.fixture(scope="module")
def ws(tmp_path_factory, tiny):
    base = tmp_path_factory.mktemp("ws")
    make_spider(base / "raw")
    return {"raw": base / "raw", "splits": base / "splits", "models": base / "models", "out": base / "results", "tiny": tiny}


def common(ws):
    return ["--raw", str(ws["raw"]), "--splits-dir", str(ws["splits"])]


def test_download_check(ws, tmp_path, capsys):
    assert download.main(["--raw", str(tmp_path)]) == 1 and "--fetch hf" in capsys.readouterr().out
    assert download.main(["--raw", str(ws["raw"])]) == 0
    facts = capsys.readouterr().out
    assert '"train_spider.json": 98' in facts and '"test_database": 1' in facts


def test_chain(ws):
    assert build_splits.main(["--raw", str(ws["raw"]), "--out", str(ws["splits"]), "--val-min", "20", "--val-max-dbs", "3"]) == 0
    s = json.loads((ws["splits"] / "summary.json").read_text())
    assert s["db_overlap_train_vs_val_dev"] == [] and s["splits"]["val"]["questions"] >= 20

    out = ["--out", str(ws["out"])]
    assert eval_model.main(["--backend", "gold", "--split", "val", *common(ws), *out]) == 0
    g = json.loads((ws["out"] / "val" / "gold.json").read_text())
    assert g["accuracy"]["ex"] == 1.0
    assert eval_model.main(["--backend", "gold", "--split", "dev", *common(ws), *out]) == 3  # closed without --final
    assert eval_model.main(["--backend", "gold", "--split", "test", *common(ws), *out]) == 3

    sft_dir = ws["models"] / "sft"
    assert train_sft.main(["--base", str(ws["tiny"]), "--out", str(sft_dir), "--device", "cpu", "--max-steps", "4", "--batch-size", "2", "--grad-accum", "1",
                           "--eval-every", "2", "--val-limit", "6", "--no-gc", "--max-len", "auto", "--results", str(ws["out"]), *common(ws)]) == 0
    assert (sft_dir / "adapter_config.json").exists() and (ws["out"] / "train_sft_sft.json").exists()
    train = load_split("train", ws["splits"])
    sft_log = json.loads((ws["out"] / "train_sft_sft.json").read_text())
    st = sft_log["train_stats"]
    assert st["kept"] + st["dropped_too_long"] == sum(e.gold_ok for e in train) and sum(e.db_empty for e in train) == 14  # SFT keeps empty databases
    assert sft_log["length_budget"]["chosen"] in (2048, 3072, 4096) and sft_log["length_budget"]["batch_size"] == 2 and sft_log["tokens_per_second"] > 0

    assert eval_model.main(["--backend", f"hf:{ws['tiny']}", "--adapter", str(sft_dir), "--split", "val", "--limit", "6", "--device", "cpu",
                            "--max-new-tokens", "12", "--tag", "tiny-sft", *common(ws), *out]) == 0
    r = json.loads((ws["out"] / "val" / "tiny-sft.json").read_text())
    assert r["accuracy"]["n"] == 6 and 0.0 <= r["accuracy"]["ex"] <= 1.0 and "usage" in r

    grpo_dir = ws["models"] / "grpo"
    assert train_grpo.main(["--base", str(ws["tiny"]), "--sft-adapter", str(sft_dir), "--out", str(grpo_dir), "--device", "cpu", "--steps", "2",
                            "--prompts-per-step", "2", "--group-size", "2", "--max-new-tokens", "8", "--eval-every", "1", "--eval-questions", "4",
                            "--micro-batch", "2", "--results", str(ws["out"]), *common(ws)]) == 0
    log = json.loads((grpo_dir / "grpo_log.json").read_text())
    assert log["steps"] == 2 and len(log["history"]) == 2 and "val_ex_before" in log and (grpo_dir / "adapter_config.json").exists()
    assert log["skipped"]["db_empty"] == 14 and log["pool"] + sum(log["skipped"].values()) == len(train)  # empty databases stay out of RL

    assert eval_model.main(["--backend", "gold", "--split", "val", "--tag", "gold2", *common(ws), *out]) == 0
    assert compare.main([str(ws["out"] / "val" / "gold.jsonl"), str(ws["out"] / "val" / "tiny-sft.jsonl"), "--out", str(ws["out"] / "cmp.json")]) == 0
    c = json.loads((ws["out"] / "cmp.json").read_text())
    assert c["n"] == 6 and c["diff"] >= 0 and 0 < c["nonempty_gold"]["n"] <= 6


def test_hf_job_command(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_FAKEtokenFAKEtokenFAKEtoken0000")
    assert run_hf_job.main(["--name", "smoke", "--commit", "abc1234", "--cmd", "python scripts/train_grpo.py --base tiny --steps 2", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--secrets HF_TOKEN --name" in out and "hf_FAKE" not in out  # the secret's name is passed, never its value
    assert "--flavor a100-large" in out and "--timeout 8h" in out and run_hf_job.IMAGE in out
    assert "tar.gz/abc1234" in out and "python scripts/train_grpo.py --base tiny --steps 2" in out
    assert "private=True" in out and '"DavutKursun/sql-tuner-job-smoke"' in out and "exit $status" in out


def test_grpo_skip_reasons():
    from dataclasses import replace

    from sqltuner.config import GRPO
    from sqltuner.data import Example

    def ex(**kw):
        return Example("x", "train", "d", "q", "SELECT 1", **kw)

    assert train_grpo.skip_reason(ex(gold_ok=False, db_empty=True)) == "gold_failed"
    assert train_grpo.skip_reason(ex(db_empty=True, gold_empty=False)) == "db_empty"  # e.g. COUNT(*) = 0 on an empty database
    assert train_grpo.skip_reason(ex(gold_empty=True)) == "gold_empty" and train_grpo.skip_reason(ex()) is None
    assert train_grpo.skip_reason(ex(db_empty=True), replace(GRPO, skip_db_empty=False)) is None


def test_kaggle_notebook(tmp_path):
    cmds = ["python scripts/train_sft.py --base X", "python scripts/eval_model.py --backend hf:X --adapter models/sft", "python scripts/eval_model.py --samples 8"]
    assert make_kaggle_notebook.main(["--name", "sft", "--user", "someone", *[x for c in cmds for x in ("--cmd", c)], "--out", str(tmp_path)]) == 0
    nb = json.loads((tmp_path / "sft" / "sql-tuner-sft.ipynb").read_text())
    meta = json.loads((tmp_path / "sft" / "kernel-metadata.json").read_text())
    src = "".join(c["source"] for c in nb["cells"])
    assert all(c in src for c in cmds) and "/kaggle/working/out" in src and "secret" not in src.lower() and "pip uninstall -y -q torchao" in src
    assert meta["id"] == "someone/sql-tuner-sft" and meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    run = nb["cells"][2]["source"].splitlines()
    assert len(run) == 3 and all(line.startswith("!") and '|| echo "' in line for line in run)  # one shell line per command, each recorded
    # the status cell: a failed command is reported, not hidden behind Kaggle's "complete"
    (tmp_path / "status").write_text("1 ok\n2 FAILED\n")
    report = nb["cells"][4]["source"].replace("/tmp/cmd_status", str(tmp_path / "status")).replace("/kaggle/working/out", str(tmp_path / "out"))
    exec(report, {})  # noqa: S102 - our own generated cell
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert [s["status"] for s in status] == ["ok", "FAILED", "not run"] and status[1]["cmd"] == cmds[1]
