# sql-tuner — a small open model that writes SQL for your database, trained with SFT + RL on execution

This file is the build spec. Read it fully before writing code. Work milestone by milestone; after each milestone
stop, show how to run it, and wait for Davut's OK. Do not spawn review or refuter sub-agents; verify with the scripts
and tests and report directly. Never print any token, key or environment-variable value (only "set: yes/no").
Run long commands in the foreground or wait on their process id; never write a polling loop that waits for a log
line. Section 0 says where we are.

## 0. Status (updated 2026-10-30)

**M3 done, start at M4.** GRPO adapter (private Hub repo `DavutKursun/sql-tuner-grpo`; a new LoRA r 16 on the merged
SFT model; 300 steps, G 8 x 4 prompts, beta 0.02, lr 5e-6, reward 1 / 0.1 / 0, max_new_tokens 128; chosen step 300 by
val-subset EX 75.0%): val EX **78.2%** greedy, **79.7%** vote-8, against SFT 74.0% / 78.7% and Claude Haiku 5.5 3-shot
85.0%. Paired GRPO − SFT +4.2 (CI by question [+2.4, +6.1], by database [+2.4, +6.8]); GRPO greedy ~ SFT vote-8
(−0.5, CI across 0); GRPO vote-8 − Haiku −5.3 (both CIs exclude 0). Easy questions back to the base level (86.0).
"No such column" halved during training; every hacking flag fell; nothing changed after the run started. Exec bonus 0
vs 0.1 (150 steps): same behaviour, +1.5 val-subset EX not significant, the bonus keeps 10 points more groups
informative. DESIGN.md §4. The val subset is part of val: M4's dev is the clean number. For M4 the model loads as
base + merge `DavutKursun/sql-tuner-sft` + adapter `DavutKursun/sql-tuner-grpo` (`eval_model.py --merge-adapter …
--adapter …`). HF Jobs still blocked (token lacks `job.write`); all GPU work on Kaggle (17.0 h used this week). 58 tests.

M2 (done): SFT adapter (`DavutKursun/sql-tuner-sft`; `max_len` 4,096, best val loss at step 200 of 996): val EX 74.0%
greedy, 78.7% vote-8; SFT − base +4.7 (CI by question excludes 0, by database does not). Of 30 SFT failures 18 are
schema linking — DESIGN.md §3. Kaggle (Oct 2026): Python 3.13, torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0
(pip keeps them; CI tests 5.16.1, 4.57.1 and latest); kernels uninstall torchao 0.10 (PEFT rejects it).

M1 (done): base model **Qwen/Qwen2.5-Coder-1.5B-Instruct** (val EX 69.4% zero-shot; Claude Haiku 5.5 3-shot 85.0%,
Sonnet 5.5 3-shot 94.0% on 300) — `eval/results/base_choice.json`, DESIGN.md §1–2. Prompt keeps 3 example rows per
table (+6.8 points on the 1.5B). Hardness port agrees 99.05 / 99.19 / 97.78% with the official script (train / val /
dev). Harness changes in M1: `db_empty`, `ex_nonempty_gold`, 200k row cap, plain greedy / sampling (no
generation_config leak), Claude 5.5 API (no temperature, thinking off), error classes.

The starter (before M1) was implemented and tested with 34 tests, offline (a miniature Spider and a tiny random model
built in code):

Implemented (`sqltuner/`):
- `config.py` (paths, split rules, base-model candidates, prompt / execution / generation / LoRA / SFT / GRPO rules,
  API prices, the system prompt)
- `db.py`: the **safe executor** — read-only SQLite (`mode=ro` + `PRAGMA query_only`), a static guard (exactly one
  SELECT / WITH statement; no writes, PRAGMA, ATTACH, load_extension), a timeout via the progress handler, a row cap;
  Spider-style result comparison (multiset unless the gold has ORDER BY, column permutations allowed, floats rounded);
  schema rendering (CREATE TABLE text + 3 example rows per table, long schemas cut table by table)
- `hardness.py`: Spider's easy / medium / hard / extra rule re-implemented on the sqlglot AST (validated in M1)
- `data.py`: locate the Spider release, build `train` / `val` / `dev` / `test`, execute every gold query once, flag
  failures and empty results, `summary.json`
- `prompts.py`: chat messages (system + schema + question, optional few-shot from OTHER databases), the SFT target,
  robust SQL extraction (code fences, unterminated fences, `<think>` blocks, trailing explanations)
- `metrics.py`: execution accuracy with **two** bootstrap CIs (by question and by database), paired differences,
  hardness breakdown, majority vote on execution results, and the RL reward with hacking monitors
- `generate.py`: backends — local Hugging Face model (+ LoRA adapter), Anthropic API (prompt caching, cost per call,
  spend cap), llama.cpp GGUF (serving), `gold` (oracle for harness tests)
- `runner.py`: evaluate a backend on a split (greedy or sample-and-vote), summaries, prediction files
- `sft.py`: LoRA SFT as a plain PyTorch loop (exact label masking, fp16 autocast + GradScaler on CUDA, gradient
  checkpointing, accumulation, cosine schedule, best-val-loss adapter kept)
- `grpo.py`: **GRPO written by hand** — group sampling, group-normalised advantages, clipped ratio, k3 KL to the
  reference (the same network with the adapter switched off), token- or sequence-level loss aggregation,
  `logits_to_keep` so only completion logits are materialised; per-step monitors (zero-variance groups, KL, length,
  executed / correct share, hacking flags, generation vs training seconds)
- `tiny.py` (offline tiny Llama + BPE tokenizer + chat template), `fixtures.py` (miniature Spider), `hub.py`
- scripts: `download.py` (`--fetch hf|gdrive`, sha256 check, safe unzip), `build_splits.py`, `eval_model.py`
  (dev / test refused without `--final`), `compare.py` (paired), `train_sft.py`, `train_grpo.py`,
  `make_kaggle_notebook.py` (kernel + metadata for `kaggle kernels push`; results come back with `kaggle kernels output`)
- tests: guard cases, read-only / timeout / row cap, result comparison, schema and extraction, hardness, splits are
  database-disjoint, CIs, reward and flags, gold scores 100%, API backend mocked (cost, cache markers, spend cap),
  SFT label masking and learning, GRPO maths, **our log-probs equal generate's own scores**, update direction, the
  whole script chain.
- NOT yet: `app.py` (FastAPI + Gradio), GGUF export, Dockerfile / Space, plots, DESIGN.md content, model card, GIF.

**Milestone 1, step 0 (do first):** Python 3.11 venv; `pip install torch`; `pip install -r requirements-dev.txt`;
`pytest -q` (expect 34 passed). Then `python scripts/download.py --fetch hf` and `python scripts/build_splits.py`;
report the facts against §3.2. Stop.

**Data finding (M1):** seven training databases have no rows at all (all of train_others + music_2; 1,759 questions):
they are flagged `db_empty`, kept for SFT and left out of the GRPO prompt pool; EX is also reported on non-empty gold
results (`ex_nonempty_gold`). Details in §3.2.

macOS notes: 8 GB RAM. **No model runs on the Mac** (rule in §2): every inference and training run is a Kaggle kernel
(2×T4, 30 GPU-hours a week); the Mac runs Claude Code, git, the Claude API baselines, tests with the tiny model and
light scripts. If Python ever "quits unexpectedly", it is the OpenMP clash solved in shop-recsys
(`scripts/fix_openmp_macos.py` there); this project does not use faiss or LightGBM, so it should not happen.

## 1. Goal

A company wants to ask its own database questions in plain language without sending its data to a hosted LLM on
every query. This project takes a small open model (0.5–1.7B parameters), teaches it SQLite with supervised
fine-tuning, then improves it with reinforcement learning whose reward is **running the SQL and comparing the result**
with the correct one, and serves it on a CPU. It measures every step honestly against the base model and against
Claude with few-shot prompting, including the cost per 1,000 questions.

The README answers: *How often does the small model return the right rows on databases it has never seen (with a
CI)?* *What did SFT add, what did RL add on top, and what does "sample 8 and vote" add for free?* *How close is it to
Claude Haiku / Sonnet, at what cost per 1,000 questions and what latency?* *What does it get wrong (hardness, error
types), and what did the RL try to cheat on?*

Portfolio purpose: modern LLM post-training (LoRA SFT, GRPO with verifiable rewards — the R1 recipe) written by hand,
reward-hacking analysis, evaluation that clusters by database, safe execution of model-written code, and a cheap
self-hosted deployment. This is the most asked-about topic in ML interviews right now.

## 2. Non-negotiables

- **Choices on val only.** val = whole Spider training databases held out (cross-database, like dev). Base model,
  prompt format, epochs, GRPO settings, the saved checkpoint: all chosen on val.
- **Spider dev is read once** (M4) for the reported numbers, and Spider test likewise if the release contains it.
  `eval_model.py --split dev|test` needs `--final`; the date and commands go into this file.
- **Execution accuracy is the metric** (right rows), with a CI by question AND by database; "A beats B" needs a
  paired CI that excludes 0 (state which level). Exact-match is not used (explain why in DESIGN.md).
- **The base model and Claude are in every table.** Base zero-shot, base few-shot, SFT, GRPO, GRPO + vote, Claude Haiku
  3-shot, Claude Sonnet 3-shot (cost-capped).
- **Model-written SQL never runs outside the safe executor** — in evaluation, in RL, and behind the demo.
- **Contamination is stated, not ignored.** Spider is public and likely in the base models' pre-training; the README
  says so, shows the base model's own score as the floor, and the optional M6 fresh set is the remedy.
- API: the key lives in `.env`; `--max-usd` caps every API run (default 5 USD); report the spend.
- **Never `git stash` or copy a working tree before a secret scan of untracked and modified files** (count
  key-shaped strings and the exact key, never print them; never write a key prefix literally into a tracked file). In M1 a stash taken while the key sat in `.env.example` left a copy in an
  unreachable git object; it was never pushed and was pruned (`git reflog expire --expire-unreachable=now --all &&
  git gc --prune=now`).
- Kaggle: credentials stay in `~/.kaggle/`; only the username is ever read. Kernels are private.
- **Every model inference and every training run happens in a Kaggle kernel** (GPU T4, internet on, private). The Mac
  only runs Claude Code, git, the Claude API baselines, tests with the tiny model, and light scripts (hardness check,
  tokenizer lengths, summaries).
- **Final read (M4) — frozen before any dev or test call** (2026-10-30). Models, prompts, decoding, checkpoints and API
  model ids as of `7ce3230`. The freeze commit adds only this entry, the kernels below, the `--seed` flag (vote-8
  sampling is seeded with 7; the val vote-8 runs were unseeded) and its test. Dev and test are each read once. Nothing
  is chosen, tuned or rerun because of a dev or test number. A rerun only after a crash or an infrastructure failure,
  named, keeping the first output. "Best of ours" for the Claude comparisons is fixed by val: GRPO vote-8 (79.7%).
  - SFT adapter: step 200 of 996. Kaggle `davutkursun/sql-tuner-sft` output `models/sft` = Hub
    `DavutKursun/sql-tuner-sft` @ `e48f343ccf295b00cf9e65d6fb4e76e1d9d5eb62`, `adapter_model.safetensors` sha256
    `5f2ef882e24e6687e3e8d2ca89d75996a87c06a43c04a34625c1d8dc63e3dea7`.
  - GRPO adapter: step 300 of 300, on the merged SFT model. Kaggle `davutkursun/sql-tuner-grpo-main` output
    `models/grpo-main/step_300` = Hub `DavutKursun/sql-tuner-grpo` @ `f3465f73747cc38761660664af60a514d8fe6dd3`, sha256
    `dc6e27fa02ed0ee03db179d55482eb7b440469df13e2b9dce3794033d41c68cf`.
  - Decoding:
    - greedy: 256 new tokens, batch 8;
    - vote-8: 8 samples at T 0.7 (no top-k / top-p / repetition penalty), majority vote on execution results, batch 4,
      seed 7.

    Prompt as on val (system + `CREATE TABLE` + 3 example rows per table + question). The 3-shot rows use the same 3
    training examples as Claude (`few_shot_pool`, seed 7).
  - API: `claude-haiku-5-5` (thinking `disabled`) and `claude-sonnet-5-5` (thinking `between_tools`), 3-shot. Estimates
    from M1's spend per question: Haiku $0.19 dev / $0.40 test, Sonnet $3.85 dev (≤ $8, so all of dev). Caps: Haiku
    dev `--max-usd 5`, Sonnet dev `--max-usd 8`, Haiku test `--max-usd 5`. Sonnet does not run on test.
  - Commands. Kaggle kernels `kaggle/final-dev` and `kaggle/final-test` (T4) run, in this order, `python
    scripts/eval_model.py --backend hf:Qwen/Qwen2.5-Coder-1.5B-Instruct --split {dev|test} --final` with:
    - `--tag Qwen2.5-Coder-1.5B-Instruct`
    - `--adapter "/kaggle/input/**/models/sft" --tag Qwen2.5-Coder-1.5B-Instruct-sft`
    - `--merge-adapter "/kaggle/input/**/models/sft" --adapter "/kaggle/input/**/models/grpo-main/step_300" --tag
      grpo-main-step300`
    - `--few-shot 3 --tag Qwen2.5-Coder-1.5B-Instruct-3shot`
    - the GRPO pair with `--samples 8 --batch-size 4 --tag grpo-main-step300-vote8`
    - the SFT adapter with `--samples 8 --batch-size 4 --tag Qwen2.5-Coder-1.5B-Instruct-sft-vote8`

    `final-dev` first fills the val row `--split val --few-shot 3`. On the Mac:
    - `python scripts/eval_model.py --backend anthropic:claude-haiku-5-5 --split dev --final --few-shot 3 --max-usd 5`
    - the same with `anthropic:claude-sonnet-5-5 … --max-usd 8`
    - `anthropic:claude-haiku-5-5 --split test … --max-usd 5`

## 3. Data

### 3.1 Source

Spider 1.0 (Yu et al., EMNLP 2018), CC BY-SA 4.0: cross-database text-to-SQL, 200 SQLite databases. Route 1: the
Hugging Face mirror `HAL-9001/spider-databases` (re-host of the official `spider_data.zip`, sha256 published) —
`python scripts/download.py --fetch hf`. Route 2: the official Google Drive file via `gdown`. The script verifies the
sha256, refuses unsafe zip paths, unzips into `data/raw/`.

### 3.2 Expected facts (verify in M1; report differences)

| Quantity | Expected | Measured in M1 (2026-10-29) |
|---|---|---|
| train_spider.json / train_others.json | 7,000 / 1,659 questions | 7,000 / 1,659 |
| dev.json | 1,034 questions on 20 databases (none shared with train) | 1,034 on 20; no database shared by any two splits |
| test.json + test_database/ | 2,147 questions on 40 databases — present only if the release includes the test set | present: 2,147 on 40 |
| gold queries that fail or return no rows | a handful; counted and reported | fail: 3 (train only). No rows: train 1,597, val 19, dev 49, test 54 — **not a handful in train**, see below |
| hardness on dev | roughly 25% easy / 40% medium / 20% hard / 15% extra | 248 / 444 / 187 / 155 (ours); the official script gives 248 / 446 / 174 / 166 (97.8% agreement) |

**Empty databases (found in M1).** Every table is empty in all six train_others databases (academic, geo, imdb,
restaurants, scholar, yelp) and in train_spider's music_2: 1,759 train questions, none in val / dev / test. 1,436 of
their gold queries return no rows and the other 323 return one row of 0 / NULL (COUNT, AVG …), so on these databases a
query that runs and returns nothing (or COUNT = 0) matches the gold result. Decisions: `build_splits` marks `db_empty`;
the GRPO prompt pool leaves these questions out (SFT keeps them; few-shot examples never come from them); summaries
report EX on all questions (Spider's convention) and `ex_nonempty_gold` on questions whose gold result has rows (val 601
of 620); the evaluation row cap is 200,000 (no gold result exceeds it; at 10,000, 16 did).

### 3.3 Splits

- train = train_spider + train_others minus val databases; val = whole train_spider databases (≥ 600 questions,
  ≤ 20 databases, very large databases skipped so val stays varied). `summary.json` records the val databases and
  checks there is no database overlap.
- Prompt: system prompt + `CREATE TABLE` text + 3 example rows per table + the question; answer in one ```sql block.

## 4. Models and training

| Step | What | Where |
|---|---|---|
| Base | candidates in `config.BASE_CANDIDATES` (Qwen2.5-Coder 0.5B / 1.5B Instruct, Qwen3 0.6B / 1.7B; Apache-2.0); chosen in M1 on val (EX, speed, size) | Mac MPS for ≤ 0.6B, Kaggle for larger |
| SFT | LoRA r 16 on all linear layers, 2 epochs, lr 2e-4, max 2,048 tokens, best val loss kept | Kaggle T4, fp16 |
| GRPO | from the merged SFT model, new LoRA, G = 8, 4 prompts/step, 300 steps, β 0.02, lr 5e-6, reward 1 / 0.1 / 0 | Kaggle T4 |
| Vote | sample 8 at T 0.7, execute, majority vote on results (test-time compute, no training) | anywhere |
| Claude | Haiku 5.5 and Sonnet 5.5, 3-shot from other databases, prompt caching | API, capped |

GRPO is the deep part: measure seconds per step first (5 steps) and size the run to fit one Kaggle session (≤ 9 h);
if too slow, reduce G or max_new_tokens before reducing steps, and say so.

## 5. Evaluation

- EX on val (all choices) and, once, on dev (+ test): CI by question and by database; by hardness; executed rate;
  error types (guard refusals, syntax, unknown column, timeout); latency p50 / p95; tokens; cost per 1,000 questions.
- Paired comparisons: SFT − base, GRPO − SFT, vote − greedy, ours − Haiku, ours − Sonnet.
- GRPO dynamics: reward, correct share, zero-variance groups, KL, completion length, flag shares per step; val EX
  every 50 steps; a section on what the policy tried to exploit (empty results, LIMIT, SELECT *, row floods) and how
  the reward or the data were changed in response (e.g. dropping gold-empty prompts).
- Error analysis: 30 failures of the final model on val, categorised by hand (schema linking, join path, aggregation,
  value mismatch, ambiguity in the question, gold arguably wrong).
- The quantised GGUF model's val EX against fp16 (M5).

## 6. Serving (M5)

- Merge the adapter, convert to GGUF with llama.cpp (`convert_hf_to_gguf.py`), quantise (Q4_K_M and Q8_0), measure val
  EX for each, serve with llama-cpp-python on CPU.
- FastAPI: `POST /ask {db, question}` → `{sql, columns, rows (≤ 200), executed, error, latency_ms, model}`;
  `POST /ask` with an uploaded SQLite file (≤ 20 MB, opened read-only, deleted after the request); `GET /databases`;
  `GET /health`. The safe executor is the only way SQL runs.
- Gradio at /ui: "Ask" (pick a Spider dev database or upload your own; question → SQL → rows; show the schema the
  model saw), "Compare" (base vs SFT vs GRPO on a saved example set), "Evidence" (tables and charts from committed
  results), "About" (data licence, contamination note, limits, nothing stored).
- Hugging Face Space, Docker, cpu-basic; latency target p50 < 3 s for a typical question at Q4.

## 7. Repository layout

```
sql-tuner/
  app.py (M5)
  sqltuner/ config, db, hardness, data, prompts, metrics, generate, runner, sft, grpo, tiny, fixtures, hub
  scripts/ download, build_splits, eval_model, compare, train_sft, train_grpo, make_kaggle_notebook, (M2+) plots,
           (M5) export_gguf, bench_latency, demo_gif
  kaggle/<name>/ (generated kernels; outputs gitignored)
  data/raw (gitignored)  data/splits (jsonl gitignored, summary.json committed)
  models/ (gitignored)  eval/results/  tests/  docs/DESIGN.md  docs/model_card.md  assets/
```

## 8. Milestones

**M1 — data, harness, baselines (val only).** Step 0; the §3.2 facts; validate `hardness.py` against the official
Spider `evaluation.py` (clone taoyds/spider, run its hardness on val + 500 train questions, report agreement; vendor it
if < 90%); prompt token lengths with each candidate's tokenizer; zero-shot val EX for the candidates (≤ 0.6B on the
Mac; ≥ 1.5B via one Kaggle kernel); Claude Haiku 3-shot on all of val and Sonnet 3-shot on 300 (cost reported);
`sample_rows` 0 vs 3 on the leading candidate; choose the base → `eval/results/base_choice.json`. DESIGN.md §1–2.
**M2 — SFT.** Kaggle kernel; curves; val EX greedy and vote-8; paired vs base; error analysis of 30 failures; push
nothing to the Hub unless asked — download the kernel output. DESIGN.md §3.
**M3 — GRPO.** Step timing; the main run; monitors and hacking findings; one ablation (β 0 vs 0.02, or exec bonus 0 vs
0.1); paired GRPO − SFT and GRPO vs vote; DESIGN.md §4.
**M4 — final.** dev once (and test once if present) for every row of §5; dev vs val table; DESIGN.md §5, model card,
README results.
**M5 — serve and ship.** GGUF, FastAPI + Gradio, upload-your-own-database, latency, Space, GIF, README.
**M6 — optional.** BIRD mini-dev (harder, dirtier databases); a fresh 100-question set on new databases written after
the base model's cutoff; DPO on SFT failures vs GRPO; a 3B model with QLoRA.

## 9. Acceptance criteria

- Harness: `gold` scores 100% minus gold failures on every split; the executor refuses writes and multi-statements
  (tests); val and dev share no database.
- SFT beats base zero-shot on val, paired CI by question excluding 0 — or the README says it does not.
- GRPO − SFT on val and dev reported with paired CIs; vote-8 reported next to it.
- Our best model vs Claude Haiku 3-shot on dev: EX with CIs, cost per 1,000 questions and latency for both.
- The quantised model loses ≤ 1.5 points of val EX against fp16 — or the README says how much it loses.
- `/ask` p50 < 3 s on the Space CPU; every response carries the SQL and whether it executed.
- `pytest` green offline; CI green; README on one screen before setup.

## 10. Out of scope

Multi-turn conversations, write queries (INSERT/UPDATE are refused by design), databases other than SQLite (Postgres
is a next step), very large schemas needing retrieval of tables (BIRD is the bridge).

## 11. Pitfalls to avoid

- Choosing anything on dev; reading dev "just to see".
- Comparing models on different questions or different prompt formats without saying so.
- Few-shot examples from the question's own database (leaks schema-specific patterns).
- Running model SQL with a writable connection, or without a timeout.
- Treating questions as independent when the claim is about new databases (use the database-level CI).
- GRPO with sampling settings (top-k / top-p) that differ from the log-prob computation; the temperature must match.
- Rewarding "executes" so much that the policy prefers safe wrong queries; watch the executed-but-wrong share.
- Printing tokens: Kaggle and HF credentials are never echoed; only the Kaggle username is read.
