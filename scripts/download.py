"""Check for the Spider release in data/raw/ and, with --fetch, download it. No credentials are read or printed.

    python scripts/download.py                 # check only; prints the two routes
    python scripts/download.py --fetch hf      # Hugging Face mirror (CC BY-SA 4.0 re-host of the official zip)
    python scripts/download.py --fetch gdrive  # the official Yale Google Drive file (needs `pip install gdown`)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import RAW, SPIDER_GDRIVE_ID, SPIDER_HF_MIRROR, SPIDER_ZIP_SHA256
from sqltuner.data import find_spider

HOW = f"""
Spider not found under {RAW}.

Spider 1.0 (Yu et al., EMNLP 2018), CC BY-SA 4.0 — ~200 MB zip with every SQLite database:
  1) python scripts/download.py --fetch hf        (mirror: huggingface.co/datasets/{SPIDER_HF_MIRROR})
  2) python scripts/download.py --fetch gdrive    (official: Google Drive id {SPIDER_GDRIVE_ID}; pip install gdown)
  or by hand: put spider_data.zip in data/raw/ and run this script again (it verifies and unzips).
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unzip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():  # refuse path traversal
            if m.startswith("/") or ".." in Path(m).parts:
                raise RuntimeError(f"unsafe path in zip: {m}")
        z.extractall(dest)


def fetch(route: str, raw: Path) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    if route == "hf":
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(SPIDER_HF_MIRROR, "spider_data.zip", repo_type="dataset", local_dir=str(raw))
        return Path(p)
    if route == "gdrive":
        import gdown

        out = raw / "spider_data.zip"
        gdown.download(id=SPIDER_GDRIVE_ID, output=str(out), quiet=False)
        return out
    raise ValueError(route)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--fetch", choices=["hf", "gdrive"], default=None)
    a = ap.parse_args(argv)
    sd = find_spider(a.raw)
    if sd is None:
        zips = list(a.raw.glob("**/spider_data.zip"))
        if not zips and a.fetch:
            zips = [fetch(a.fetch, a.raw)]
        if not zips:
            print(HOW)
            return 1
        digest = sha256(zips[0])
        print(f"zip {zips[0].name}: {zips[0].stat().st_size / 1e6:.1f} MB, sha256 {'matches the published value' if digest == SPIDER_ZIP_SHA256 else 'DIFFERS: ' + digest}")
        unzip(zips[0], a.raw)
        sd = find_spider(a.raw)
        if sd is None:
            print("unzipped, but no train_spider.json + database/ found", file=sys.stderr)
            return 2
    facts = {"folder": str(sd.relative_to(a.raw.parent) if a.raw.parent in sd.parents else sd)}
    for name in ("train_spider.json", "train_others.json", "dev.json", "test.json", "tables.json", "test_tables.json"):
        p = sd / name
        facts[name] = len(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None
    for folder in ("database", "test_database"):
        p = sd / folder
        facts[folder] = len([d for d in p.iterdir() if d.is_dir()]) if p.exists() else None
    print(json.dumps(facts, indent=2))
    if facts["test.json"] is None or facts["test_database"] is None:
        print("note: no Spider test set in this release; the protocol reports Spider dev once (M4) and keeps val for every choice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
