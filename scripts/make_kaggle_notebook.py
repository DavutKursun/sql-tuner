"""Write a thin Kaggle kernel (notebook + kernel-metadata.json) that clones this repo into /tmp, installs the training
deps, fetches Spider, runs the given commands in order (a failing one does not stop the next), and copies models/ and
eval/results/ into /kaggle/working (the kernel output). No secrets are needed: the trained adapter comes back with
`kaggle kernels output`.

A failing shell command does not fail a notebook, so Kaggle says "complete" either way: each command's outcome is
written to out/status.json (and printed last) — read it after downloading the output.

    python scripts/make_kaggle_notebook.py --user <kaggle username> --name sft \
        --cmd "python scripts/train_sft.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --out models/sft" --cmd "python scripts/eval_model.py ..."
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
STATUS = "/tmp/cmd_status"


def cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cmd", required=True, action="append", help="a command run from the repo root; repeat for several")
    ap.add_argument("--user", default="USERNAME", help="Kaggle username (only the name; never the key)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle")
    a = ap.parse_args(argv)
    run = "\n".join(f'!{c} && echo "{i} ok" >> {STATUS} || echo "{i} FAILED" >> {STATUS}' for i, c in enumerate(a.cmd, 1))
    report = (f"import json, pathlib\ncmds = {json.dumps(a.cmd)}\n"
              f"done = dict(line.split(' ', 1) for line in pathlib.Path('{STATUS}').read_text().split('\\n') if line) if pathlib.Path('{STATUS}').exists() else {{}}\n"
              "status = [{'cmd': c, 'status': done.get(str(i), 'not run')} for i, c in enumerate(cmds, 1)]\n"
              "pathlib.Path('/kaggle/working/out').mkdir(parents=True, exist_ok=True)\n"
              "pathlib.Path('/kaggle/working/out/status.json').write_text(json.dumps(status, indent=1))\n"
              "print('COMMANDS:', [s['status'] for s in status])")
    cells = [
        # Kaggle's image ships torchao 0.10, which PEFT >= 0.17 rejects as soon as it builds LoRA layers; nothing here uses it
        cell(f"!git clone --depth 1 -b {a.branch} {REPO} /tmp/sql-tuner\n%cd /tmp/sql-tuner\n!git log -1 --format='commit %h %s'\n"
             "!pip install -q -r requirements-train.txt && pip uninstall -y -q torchao\n!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
             "!python -c \"import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)\""),
        cell("!python scripts/download.py --fetch hf && python scripts/build_splits.py > /tmp/splits.log && tail -5 /tmp/splits.log"),
        cell(run),
        cell("!mkdir -p /kaggle/working/out && cp -r models eval /kaggle/working/out/ && du -sh /kaggle/working/out/*"),
        cell(report),
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
