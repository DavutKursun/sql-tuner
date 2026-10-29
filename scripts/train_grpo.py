"""GRPO with an execution reward, starting from the SFT model merged into the base, a fresh LoRA on top (dropout 0, so
the policy that samples and the policy that is scored are the same function), and that merged SFT model as the KL
reference (the new adapter switched off). Greedy EX on a fixed val subset at step 0 and every eval_every steps; every
one of those checkpoints is saved, and the best (ties: the earlier step) is also kept in --out itself.

    python scripts/train_grpo.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct --sft-adapter "/kaggle/input/**/models/sft" \
        --settings-from eval/results/grpo_probe.json --val-subset eval/results/val_subset_ids.json --max-hours 8 --out models/grpo-main
    python scripts/train_grpo.py --base tiny --steps 2 --group-size 2 --out /tmp/grpo     # offline smoke run

Prompt pool: train only, without gold failures, empty gold results (returning nothing would win there), databases with
no rows, and prompts longer than --max-prompt-tokens. --settings-from takes max_new_tokens and the hardness weights from
scripts/grpo_probe.py's output; flags given explicitly win.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # long prompts fragment the T4's 15 GB otherwise
sys.path.insert(0, str(ROOT))

from sqltuner.config import GEN, GRPO, LORA, MODELS, PROMPT, RAW, RESULTS, SPLITS
from sqltuner.data import db_path, find_spider, load_split, stratified_subset
from sqltuner.db import ExecResult, Executor
from sqltuner.generate import HFBackend, find_adapter, load_model
from sqltuner.grpo import train_grpo
from sqltuner.metrics import reward
from sqltuner.prompts import apply_chat_template, extract_sql, messages, schema_text
from sqltuner.runner import evaluate, write_predictions
from sqltuner.sft import add_lora

sys.path.insert(0, str(ROOT / "scripts"))
from train_sft import resolve_base


def skip_reason(e, rules=GRPO) -> str | None:
    """Why a training example stays out of the prompt pool (None = it is used); prompt length is checked separately."""
    if not e.gold_ok:
        return "gold_failed"
    if e.db_empty and rules.skip_db_empty:
        return "db_empty"
    if e.gold_empty and rules.skip_gold_empty:
        return "gold_empty"
    return None


def build_pool(tok, spider_dir: Path, splits_dir: Path, rules=GRPO, max_prompt_tokens: int = 2048):
    """(pool examples, their chat-formatted prompts, skipped counts by reason) from the train split."""
    pool, prompts, skipped = [], [], {"gold_failed": 0, "db_empty": 0, "gold_empty": 0, "too_long": 0}
    for e in load_split("train", splits_dir):
        why = skip_reason(e, rules)
        if why:
            skipped[why] += 1
            continue
        sch = schema_text(str(db_path(spider_dir, e.db_id, e.split)), PROMPT.sample_rows, PROMPT.max_cell_chars, PROMPT.max_schema_chars)
        text = apply_chat_template(tok, messages(sch, e.question))
        if len(tok(text, add_special_tokens=False)["input_ids"]) > max_prompt_tokens:
            skipped["too_long"] += 1
            continue
        pool.append(e)
        prompts.append(text)
    return pool, prompts, skipped


def hardness_mix(examples) -> dict:
    c = Counter(e.hardness for e in examples)
    return {h: {"n": c[h], "share": round(c[h] / max(len(examples), 1), 4)} for h in ("easy", "medium", "hard", "extra", "unknown") if c[h]}


def pool_weights(pool, hardness_weights: dict | None) -> np.ndarray | None:
    """Per-prompt sampling probabilities: each hardness gets its weight, split evenly over its prompts (None: uniform)."""
    if not hardness_weights:
        return None
    counts = Counter(e.hardness for e in pool)
    p = np.array([hardness_weights.get(e.hardness, 0.0) / counts[e.hardness] for e in pool], dtype=float)
    return p / p.sum()


def make_reward_fn(pool, spider_dir: Path, executor: Executor, exec_bonus: float):
    def reward_fn(i: int, text: str):
        e = pool[i]
        path = db_path(spider_dir, e.db_id, e.split)
        sql = extract_sql(text)
        r = reward(sql, executor.run(path, sql) if sql else ExecResult(False, error="no sql"), executor.gold(path, e.query), e.query, exec_bonus)
        return r.value, {"correct": r.correct, "executed": r.executed, "flags": r.flags, "failure": r.failure, "sql": sql}

    return reward_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft-adapter", default=None, help="folder, glob (Kaggle input) or Hub repo; merged into the base before GRPO")
    ap.add_argument("--out", type=Path, default=MODELS / "grpo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=GRPO.steps)
    ap.add_argument("--prompts-per-step", type=int, default=GRPO.prompts_per_step)
    ap.add_argument("--group-size", type=int, default=GRPO.group_size)
    ap.add_argument("--temperature", type=float, default=GRPO.temperature)
    ap.add_argument("--max-new-tokens", type=int, default=None, help=f"default: --settings-from, else {GRPO.max_new_tokens}")
    ap.add_argument("--lr", type=float, default=GRPO.lr)
    ap.add_argument("--beta", type=float, default=GRPO.beta)
    ap.add_argument("--inner-epochs", type=int, default=GRPO.inner_epochs)
    ap.add_argument("--loss-agg", choices=["token", "sequence"], default=GRPO.loss_agg)
    ap.add_argument("--micro-batch", type=int, default=GRPO.micro_batch)
    ap.add_argument("--exec-bonus", type=float, default=GRPO.exec_bonus)
    ap.add_argument("--keep-gold-empty", action="store_true")
    ap.add_argument("--keep-db-empty", action="store_true", help="keep questions on databases whose tables are all empty")
    ap.add_argument("--eval-every", type=int, default=GRPO.eval_every)
    ap.add_argument("--eval-questions", type=int, default=GRPO.eval_questions, help="size of the stratified val subset when --val-subset is not given")
    ap.add_argument("--val-subset", type=Path, default=None, help="JSON with the ids of the fixed val subset")
    ap.add_argument("--hardness-weights", default=None, help="JSON {hardness: weight}, or 'uniform'; default: --settings-from, else uniform")
    ap.add_argument("--settings-from", type=Path, default=None, help="scripts/grpo_probe.py output")
    ap.add_argument("--max-hours", type=float, default=None, help="wall-clock guard for the training loop")
    ap.add_argument("--lora-r", type=int, default=LORA.r)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048, help="training prompts longer than this are skipped")
    ap.add_argument("--keep-examples", type=int, default=6, help="completions kept per hacking flag")
    ap.add_argument("--no-gc", action="store_true", help="no gradient checkpointing in the policy pass")
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
    settings = json.loads(a.settings_from.read_text()) if a.settings_from else {}
    max_new = a.max_new_tokens or settings.get("max_new_tokens") or GRPO.max_new_tokens
    hw = None if a.hardness_weights == "uniform" else (json.loads(a.hardness_weights) if a.hardness_weights else settings.get("hardness_weights"))
    base = resolve_base(a.base)
    tok, model, device = load_model(base, device=a.device, trainable=True, merge=find_adapter(a.sft_adapter))
    model = add_lora(model, replace(LORA, r=a.lora_r, alpha=2 * a.lora_r, dropout=0.0))
    if not a.no_gc and device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    rules = replace(GRPO, steps=a.steps, prompts_per_step=a.prompts_per_step, group_size=a.group_size, temperature=a.temperature, max_new_tokens=max_new,
                    lr=a.lr, beta=a.beta, inner_epochs=a.inner_epochs, loss_agg=a.loss_agg, micro_batch=a.micro_batch, exec_bonus=a.exec_bonus,
                    skip_db_empty=not a.keep_db_empty, skip_gold_empty=not a.keep_gold_empty, eval_every=a.eval_every, eval_questions=a.eval_questions)
    ex = Executor()
    pool, prompts, skipped = build_pool(tok, sd, a.splits_dir, rules, a.max_prompt_tokens)
    mix = hardness_mix(pool)
    weights = pool_weights(pool, hw)
    print(f"prompt pool {len(prompts):,}; skipped {skipped}; hardness {mix}")
    print(f"settings: max_new_tokens {max_new}, prompt sampling {'by hardness ' + json.dumps(hw) if hw else 'uniform'}, exec bonus {rules.exec_bonus}, "
          f"G {rules.group_size} x {rules.prompts_per_step} prompts, beta {rules.beta}, lr {rules.lr}")

    val_all = load_split("val", a.splits_dir)
    if a.val_subset:
        ids = set(json.loads(a.val_subset.read_text())["ids"])
        val = [e for e in val_all if e.id in ids]
    else:
        val = stratified_subset(val_all, rules.eval_questions)
    tag = a.tag or a.out.name
    backend = HFBackend("policy", tok=tok, model=model, device=device, batch_size=8)
    subset_dir = a.results / "val_subset" / tag

    def eval_fn(step: int) -> float:
        tok.padding_side = "left"
        preds, _ = evaluate(val, backend, sd, PROMPT, (), 1, max_new_tokens=GEN.max_new_tokens, log=None)  # the full-val evaluation's budget
        write_predictions(preds, subset_dir / f"step_{step:03d}.jsonl")
        model.train()
        return float(np.mean([p.correct for p in preds]))

    info = train_grpo(model, tok, prompts, make_reward_fn(pool, sd, ex, rules.exec_bonus), a.out, device, rules, eval_fn=eval_fn, weights=weights,
                      max_hours=a.max_hours, prompt_ids=[e.id for e in pool], keep_examples=a.keep_examples)
    by_id = {e.id: e for e in pool}
    for exs in info["examples"].values():
        for x in exs:
            x["question"], x["gold"], x["db_id"] = by_id[x["prompt"]].question, by_id[x["prompt"]].query, by_id[x["prompt"]].db_id
    info.update({"base": a.base, "sft_adapter": a.sft_adapter, "pool": len(prompts), "skipped": skipped, "pool_hardness": mix, "hardness_weights": hw,
                 "max_prompt_tokens": a.max_prompt_tokens, "val_subset": [e.id for e in val], "device": device, "settings_from": str(a.settings_from or ""),
                 "gradient_checkpointing": not a.no_gc and device == "cuda", "lora_dropout": 0.0})
    (a.out / "grpo_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"train_grpo_{tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"val-subset EX by step: {[(e['step'], round(e['val_ex'], 4)) for e in info['evals']]}; best step {info['best_step']}; "
          f"{info['steps']} steps in {info['minutes']} min{'; ' + info['stopped'] if info['stopped'] else ''}")
    ex.close()
    if a.push_to:
        from sqltuner.hub import push_folder

        print("pushed:", push_folder(a.out, a.push_to, message=f"GRPO adapter, val-subset EX {info['best_val_ex']:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
