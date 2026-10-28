"""GRPO with an execution reward, starting from the SFT model (merged), a fresh LoRA on top, the SFT model as the
KL reference (adapter switched off). Greedy val EX on a fixed subset picks the saved adapter.

    python scripts/train_grpo.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --sft-adapter DavutKursun/sql-tuner-sft --out models/grpo
    python scripts/train_grpo.py --base tiny --steps 2 --group-size 2 --out /tmp/grpo     # offline smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqltuner.config import GRPO, LORA, MODELS, PROMPT, RAW, RESULTS, SEED, SPLITS
from sqltuner.data import db_path, find_spider, load_split
from sqltuner.db import ExecResult, Executor
from sqltuner.generate import HFBackend, load_model
from sqltuner.grpo import train_grpo
from sqltuner.metrics import reward
from sqltuner.prompts import apply_chat_template, extract_sql, messages, schema_text
from sqltuner.runner import evaluate
from sqltuner.sft import add_lora

sys.path.insert(0, str(ROOT / "scripts"))
from train_sft import resolve_base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft-adapter", default=None, help="local folder or Hub repo; merged into the base before GRPO")
    ap.add_argument("--out", type=Path, default=MODELS / "grpo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=GRPO.steps)
    ap.add_argument("--prompts-per-step", type=int, default=GRPO.prompts_per_step)
    ap.add_argument("--group-size", type=int, default=GRPO.group_size)
    ap.add_argument("--temperature", type=float, default=GRPO.temperature)
    ap.add_argument("--max-new-tokens", type=int, default=GRPO.max_new_tokens)
    ap.add_argument("--lr", type=float, default=GRPO.lr)
    ap.add_argument("--beta", type=float, default=GRPO.beta)
    ap.add_argument("--inner-epochs", type=int, default=GRPO.inner_epochs)
    ap.add_argument("--loss-agg", choices=["token", "sequence"], default=GRPO.loss_agg)
    ap.add_argument("--micro-batch", type=int, default=GRPO.micro_batch)
    ap.add_argument("--exec-bonus", type=float, default=GRPO.exec_bonus)
    ap.add_argument("--keep-gold-empty", action="store_true")
    ap.add_argument("--eval-every", type=int, default=GRPO.eval_every)
    ap.add_argument("--eval-questions", type=int, default=GRPO.eval_questions)
    ap.add_argument("--lora-r", type=int, default=LORA.r)
    ap.add_argument("--max-prompt-tokens", type=int, default=1800, help="training prompts longer than this are skipped")
    ap.add_argument("--push-to", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    sd = find_spider(a.raw)
    if sd is None:
        print("Spider not found; run scripts/download.py", file=sys.stderr)
        return 1
    base = resolve_base(a.base)
    tok, model, device = load_model(base, device=a.device, trainable=True)
    if a.sft_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.sft_adapter).merge_and_unload()
    model = add_lora(model, replace(LORA, r=a.lora_r, alpha=2 * a.lora_r))
    rules = replace(GRPO, steps=a.steps, prompts_per_step=a.prompts_per_step, group_size=a.group_size, temperature=a.temperature, max_new_tokens=a.max_new_tokens,
                    lr=a.lr, beta=a.beta, inner_epochs=a.inner_epochs, loss_agg=a.loss_agg, micro_batch=a.micro_batch, exec_bonus=a.exec_bonus,
                    skip_gold_empty=not a.keep_gold_empty, eval_every=a.eval_every, eval_questions=a.eval_questions)
    ex = Executor()
    pool_ex, prompts, skipped = [], [], {"gold_failed": 0, "gold_empty": 0, "too_long": 0}
    for e in load_split("train", a.splits_dir):
        if not e.gold_ok:
            skipped["gold_failed"] += 1
            continue
        if e.gold_empty and rules.skip_gold_empty:
            skipped["gold_empty"] += 1
            continue
        sch = schema_text(str(db_path(sd, e.db_id, e.split)), PROMPT.sample_rows, PROMPT.max_cell_chars, PROMPT.max_schema_chars)
        text = apply_chat_template(tok, messages(sch, e.question))
        if len(tok(text, add_special_tokens=False)["input_ids"]) > a.max_prompt_tokens:
            skipped["too_long"] += 1
            continue
        pool_ex.append(e)
        prompts.append(text)
    print(f"prompt pool {len(prompts):,}; skipped {skipped}")

    def reward_fn(i: int, text: str):
        e = pool_ex[i]
        path = db_path(sd, e.db_id, e.split)
        sql = extract_sql(text)
        r = reward(sql, ex.run(path, sql) if sql else ExecResult(False, error="no sql"), ex.gold(path, e.query), e.query, rules.exec_bonus)
        return r.value, {"correct": r.correct, "executed": r.executed, "flags": r.flags}

    val = load_split("val", a.splits_dir)
    val = [val[i] for i in np.sort(np.random.default_rng(SEED).choice(len(val), min(rules.eval_questions, len(val)), replace=False))]
    backend = HFBackend("policy", tok=tok, model=model, device=device, batch_size=8)

    def eval_fn() -> float:
        tok.padding_side = "left"
        preds, _ = evaluate(val, backend, sd, PROMPT, (), 1, max_new_tokens=rules.max_new_tokens, log=None)
        model.train()
        return float(np.mean([p.correct for p in preds]))

    v0 = eval_fn()
    print(f"val EX before GRPO on {len(val)} questions: {v0:.4f}")
    info = train_grpo(model, tok, prompts, reward_fn, a.out, device, rules, eval_fn=eval_fn)
    info.update({"base": a.base, "sft_adapter": a.sft_adapter, "val_ex_before": v0, "pool": len(prompts), "skipped": skipped, "device": device})
    (a.out / "grpo_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    tag = a.tag or a.out.name
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"train_grpo_{tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"val EX {v0:.4f} → best {info['best_val_ex']:.4f}; {info['minutes']} min; adapter in {a.out}")
    ex.close()
    if a.push_to:
        from sqltuner.hub import push_folder

        print("pushed:", push_folder(a.out, a.push_to, message=f"GRPO adapter, val EX {info['best_val_ex']:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
