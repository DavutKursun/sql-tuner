# Design notes

Filled milestone by milestone; each section says what was measured, on which split, and what it changes.

## 1. Data, splits and the harness (M1)
Spider facts vs the published ones, the val databases, gold failures and empty results, hardness distribution and
the check of our hardness port against the official script, prompt lengths per tokenizer, the safe executor and why
execution accuracy (not exact match).

## 2. Baselines on val (M1)
Base candidates zero-shot, Claude Haiku / Sonnet 3-shot with cost, example rows in the prompt or not, the base choice.

## 3. Supervised fine-tuning (M2)
Label masking, LoRA settings, curves, val EX greedy and vote-8, paired vs base, 30 failures categorised.

## 4. Reinforcement learning with an execution reward (M3)
The GRPO objective as implemented, step timing on a T4, the run, monitors (zero-variance groups, KL, length,
executed-but-wrong share), what the policy tried to exploit and what changed in response, the ablation, GRPO vs vote.

## 5. Final results (M4) — dev read once on _(date)_

## 6. What I would do differently
