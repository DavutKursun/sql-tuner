"""Before GRPO: how often does the SFT model get a training prompt right when it samples exactly as GRPO will (T 1.0,
no top-k / top-p, the settings the log-probs assume), and how long are its completions? Decides how training prompts
are drawn and max_new_tokens.

    python scripts/grpo_probe.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --sft-adapter "/kaggle/input/**/models/sft" \
        --out eval/results/grpo_probe.json

--prompts random pool prompts, --samples completions each, every completion executed and graded. A group whose samples
are all right or all wrong (0/8 or 8/8) has no reward variance and teaches GRPO nothing. If such groups are more than
half, prompts are drawn by hardness in proportion to each hardness's share of the mixed groups; otherwise uniformly.
max_new_tokens = p99 of the completion lengths rounded up to a multiple of 32, capped at 256.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # long prompts fragment the T4's 15 GB otherwise
sys.path.insert(0, str(ROOT))

from sqltuner.config import GRPO, RAW, SEED, SPLITS
from sqltuner.data import db_path, find_spider
from sqltuner.db import ExecResult, Executor
from sqltuner.generate import find_adapter, load_model
from sqltuner.grpo import sample_groups
from sqltuner.metrics import reward
from sqltuner.prompts import extract_sql

sys.path.insert(0, str(ROOT / "scripts"))
from train_grpo import build_pool
from train_sft import resolve_base

LEVELS = ("easy", "medium", "hard", "extra")


def decide(per_prompt: list[dict], G: int) -> dict:
    """Zero-variance share (0/G or G/G correct) and the prompt-sampling rule it implies."""
    zero = [p["correct"] in (0, G) for p in per_prompt]
    by_h: dict = {}
    for h in (*LEVELS, "unknown"):
        rows = [p for p in per_prompt if p["hardness"] == h]
        if rows:
            ks = [p["correct"] for p in rows]
            by_h[h] = {"n": len(rows), "correct_of_G": {str(k): ks.count(k) for k in range(G + 1)}, "correct_rate": round(float(np.mean(ks)) / G, 4),
                       "zero_variance_share": round(float(np.mean([k in (0, G) for k in ks])), 4), "mixed": sum(0 < k < G for k in ks)}
    mixed_total = sum(v["mixed"] for v in by_h.values())
    share = float(np.mean(zero)) if zero else 0.0
    weights = {h: round(v["mixed"] / mixed_total, 4) for h, v in by_h.items()} if share > 0.5 and mixed_total else None
    return {"zero_variance_share": round(share, 4), "by_hardness": by_h, "decision": "by hardness" if weights else "uniform", "hardness_weights": weights}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--samples", type=int, default=GRPO.group_size)
    ap.add_argument("--temperature", type=float, default=GRPO.temperature)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="the probe's own cap (and the cap of the recommendation)")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--batch-prompts", type=int, default=4, help="prompts per generate call (x samples sequences)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    a = ap.parse_args(argv)
    sd = find_spider(a.raw)
    if sd is None:
        print("Spider not found; run scripts/download.py", file=sys.stderr)
        return 1
    import torch

    torch.manual_seed(a.seed)
    tok, model, device = load_model(resolve_base(a.base), device=a.device, merge=find_adapter(a.sft_adapter))
    pool, prompts, skipped = build_pool(tok, sd, a.splits_dir, GRPO, a.max_prompt_tokens)
    pick = np.random.default_rng(a.seed).choice(len(pool), size=min(a.prompts, len(pool)), replace=False)
    ex = Executor()
    G, per_prompt, lengths, cut_all = a.samples, [], [], []
    t0 = time.time()
    for s in range(0, len(pick), a.batch_prompts):
        idx = [int(i) for i in pick[s : s + a.batch_prompts]]
        smp = sample_groups(model, tok, [prompts[i] for i in idx], G, a.temperature, a.max_new_tokens, device)
        lens, cut = smp.comp_mask.sum(1).cpu().numpy(), smp.truncated
        for k, i in enumerate(idx):
            e = pool[i]
            path = db_path(sd, e.db_id, e.split)
            correct = executed = 0
            for j in range(k * G, (k + 1) * G):
                sql = extract_sql(smp.texts[j])
                r = reward(sql, ex.run(path, sql) if sql else ExecResult(False, error="no sql"), ex.gold(path, e.query), e.query, GRPO.exec_bonus)
                correct += int(r.correct and not cut[j])
                executed += int(r.executed and not cut[j])
            per_prompt.append({"id": e.id, "hardness": e.hardness, "correct": correct, "executed": executed})
        lengths.extend(int(x) for x in lens)
        cut_all.extend(bool(x) for x in cut)
        print(f"  {len(per_prompt)}/{len(pick)} prompts  mean correct {np.mean([p['correct'] for p in per_prompt]) / G:.3f}  ({time.time() - t0:.0f} s)")
    ex.close()
    L = np.asarray(lengths)
    p50, p95, p99 = (float(np.percentile(L, q)) for q in (50, 95, 99))
    rec = min(256, int(math.ceil(p99 / 32) * 32))
    out = {"prompts": len(per_prompt), "samples": G, "temperature": a.temperature, "top_k": 0, "top_p": 1.0, "probe_max_new_tokens": a.max_new_tokens,
           "pool": len(pool), "skipped": skipped, **decide(per_prompt, G),
           "lengths": {"p50": p50, "p95": p95, "p99": p99, "max": int(L.max()), "truncated_at_probe_cap": round(float(np.mean(cut_all)), 4)},
           "max_new_tokens": rec, "would_truncate_at_max_new_tokens": round(float(np.mean(L > rec)), 4), "seconds": round(time.time() - t0, 1),
           "per_prompt": per_prompt}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_prompt",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
