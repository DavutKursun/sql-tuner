"""Write a thin Kaggle kernel (notebook + kernel-metadata.json) that clones this repo into /tmp, installs the training
deps, fetches Spider, runs ONE command, and copies models/ and eval/results/ into /kaggle/working (the kernel output).
No secrets are needed: the trained adapter comes back with `kaggle kernels output`.

    python scripts/make_kaggle_notebook.py --user <kaggle username> --name sft \
        --cmd "python scripts/train_sft.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --out models/sft"
    kaggle kernels push -p kaggle/sft --accelerator NvidiaTeslaT4   # starts the run on T4 x2 (GPU and internet on)
    kaggle kernels status <user>/sql-tuner-sft                 # queued / running / complete / error
    kaggle kernels output <user>/sql-tuner-sft -p kaggle/sft/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/DavutKursun/sql-tuner"


def cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cmd", required=True, help="one command, run from the repo root")
    ap.add_argument("--user", default="USERNAME", help="Kaggle username (only the name; never the key)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle")
    a = ap.parse_args(argv)
    cells = [
        cell(f"!git clone --depth 1 -b {a.branch} {REPO} /tmp/sql-tuner\n%cd /tmp/sql-tuner\n!git log -1 --format='commit %h %s'\n"
             "!pip install -q -r requirements-train.txt\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        cell("!python scripts/download.py --fetch hf && python scripts/build_splits.py > /tmp/splits.log && tail -5 /tmp/splits.log"),
        cell(f"!{a.cmd}"),
        cell("!mkdir -p /kaggle/working/out && cp -r models eval /kaggle/working/out/ && du -sh /kaggle/working/out/*"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    folder = a.out / a.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"sql-tuner-{a.name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{a.user}/sql-tuner-{a.name}", "title": f"sql-tuner-{a.name}", "code_file": f"sql-tuner-{a.name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
