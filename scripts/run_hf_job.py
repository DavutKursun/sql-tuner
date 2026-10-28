"""Run ONE command on Hugging Face Jobs (GPU billed by the minute) and push its results to a private Hub repo.

The job starts from a stock PyTorch image, downloads this repository at a given commit from GitHub (a tarball, so the
image needs no git), installs requirements-train.txt, fetches Spider, builds the splits, runs the command, then uploads
models/ and eval/results/ to a private model repo — also when the command fails, so a crash still leaves its logs and
partial results. The Hub token reaches the job only as a secret (`--secrets HF_TOKEN`, taken from your environment by
the `hf` CLI); nothing here reads or prints it.

    python scripts/run_hf_job.py --name grpo --cmd "python scripts/train_grpo.py ..."          # a100-large, 8 h timeout
    python scripts/run_hf_job.py --name smoke --flavor t4-small --timeout 10m --wait \
        --cmd "python scripts/train_grpo.py --base tiny --steps 2 --group-size 2 --prompts-per-step 2 --out models/grpo"
    hf jobs logs <job id>     hf jobs inspect <job id>     hf jobs cancel <job id>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "DavutKursun/sql-tuner"
IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"  # torch >= 2.6 (transformers 5 needs it), CUDA 12.6 runs on T4 and A100

SCRIPT = r"""set -eo pipefail
cd /tmp
python - <<'PY'
import io, tarfile, urllib.request
data = urllib.request.urlopen("https://codeload.github.com/__GITHUB__/tar.gz/__COMMIT__", timeout=300).read()
archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
try:
    archive.extractall("/tmp", filter="data")
except TypeError:  # Python without extraction filters
    archive.extractall("/tmp")
PY
cd "/tmp/$(basename __GITHUB__)-__COMMIT__"
echo "commit __COMMIT__"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
pip install -q --root-user-action=ignore -r requirements-train.txt
python scripts/download.py --fetch hf > /tmp/download.log && tail -3 /tmp/download.log
python scripts/build_splits.py > /tmp/splits.log && tail -4 /tmp/splits.log
set +e
__CMD__
status=$?
set -e
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
api, repo = HfApi(), "__RESULTS__"
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
for folder in ("models", "eval/results"):
    if any(p.is_file() and p.name != ".gitkeep" for p in Path(folder).rglob("*")):
        api.upload_folder(folder_path=folder, path_in_repo=folder, repo_id=repo, repo_type="model", commit_message=f"job results: {folder}")
print("results pushed to", repo)
PY
echo "command exit status $status"
exit $status
"""


def job_script(commit: str, cmd: str, results_repo: str) -> str:
    return SCRIPT.replace("__GITHUB__", GITHUB).replace("__COMMIT__", commit).replace("__RESULTS__", results_repo).replace("__CMD__", cmd)


def hf_cli() -> str:
    exe = Path(sys.executable).with_name("hf")
    return str(exe) if exe.exists() else (shutil.which("hf") or "hf")


def build(a, commit: str) -> list[str]:
    return [hf_cli(), "jobs", "run", "--flavor", a.flavor, "--timeout", a.timeout, "--secrets", "HF_TOKEN", "--name", f"sql-tuner-{a.name}",
            "--detach", "--format", "json", a.image, "bash", "-c", job_script(commit, a.cmd, a.results_repo or f"DavutKursun/sql-tuner-job-{a.name}")]


def price_per_minute(flavor: str) -> float | None:
    out = subprocess.run([hf_cli(), "jobs", "hardware"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        cols = line.split("\t")
        if cols and cols[0] == flavor and len(cols) >= 7 and cols[6].startswith("$"):
            return float(cols[6][1:])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short job name; results go to DavutKursun/sql-tuner-job-<name> unless --results-repo")
    ap.add_argument("--cmd", required=True, help="one command, run from the repo root")
    ap.add_argument("--flavor", default="a100-large", help="see `hf jobs hardware`")
    ap.add_argument("--timeout", default="8h", help="e.g. 10m, 8h")
    ap.add_argument("--commit", default=None, help="default: HEAD, which must be pushed to GitHub")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--results-repo", default=None)
    ap.add_argument("--wait", action="store_true", help="block until the job ends, then report minutes and cost")
    ap.add_argument("--dry-run", action="store_true", help="print the job script; launch nothing")
    a = ap.parse_args(argv)
    commit = a.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    cmd = build(a, commit)
    if a.dry_run:
        print(" ".join(cmd[:-1]), "<script below>\n" + cmd[-1])
        return 0
    on_remote = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    if not on_remote:
        print(f"commit {commit[:7]} is not on GitHub yet: push it first (the job downloads it from there)", file=sys.stderr)
        return 2
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        print(out.stderr.strip()[-2000:], file=sys.stderr)
        return out.returncode
    try:
        job_id = json.loads(out.stdout)["id"]
    except (ValueError, KeyError, TypeError):
        m = re.search(r"\b[0-9a-f]{24}\b", out.stdout)  # Hub job ids are 24 hex characters
        job_id = m.group(0) if m else out.stdout.strip().split()[-1]
    print(f"job {job_id} on {a.flavor} (commit {commit[:7]}, timeout {a.timeout}); follow with: hf jobs logs {job_id}")
    if not a.wait:
        return 0
    subprocess.run([hf_cli(), "jobs", "wait", job_id], capture_output=True, check=False)
    info = json.loads(subprocess.run([hf_cli(), "jobs", "inspect", job_id, "--json"], capture_output=True, text=True, check=False).stdout)
    info = info[0] if isinstance(info, list) else info
    print(json.dumps({k: info.get(k) for k in ("id", "status", "createdAt", "startedAt", "finishedAt", "flavor")}, indent=1, default=str))
    rate = price_per_minute(a.flavor)
    print(f"price {a.flavor}: ${rate}/min" if rate is not None else "price unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
