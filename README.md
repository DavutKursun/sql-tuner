# sql-tuner

**A small open model that turns plain-language questions into SQL for your own database — fine-tuned, then trained
with reinforcement learning on whether its query returns the right rows, and served on a CPU.**

_(demo GIF and live link arrive in Milestone 5)_

## The problem
Teams want to ask their database questions without writing SQL and without sending their schema and data to a hosted
LLM on every query. Small open models are cheap and private but make more mistakes; the question is how much training
closes the gap, and how to measure it honestly.

## What this does
- Fine-tunes a 0.5–1.7B open model with LoRA on Spider, then improves it with GRPO (the DeepSeek-R1 recipe) whose
  reward is executing the SQL and comparing the result with the correct one — both written by hand in PyTorch
- Measures execution accuracy on databases the model has never seen, with confidence intervals by question and by
  database, against the base model and against Claude with few-shot prompting, including cost per 1,000 questions
- Runs every model-written query in a read-only, time-limited, single-statement sandbox, and reports what the RL
  policy tried to exploit
- Serves the quantised model with llama.cpp on a free CPU, on Spider databases or a SQLite file you upload

## Results
_(filled at Milestone 4 — Spider dev, read once)_

| Model | Execution accuracy (95% CI) | Cost per 1,000 questions | Latency p50 |
|---|---|---|---|
| Base model, zero-shot | | | |
| + SFT | | | |
| + GRPO | | | |
| + GRPO, sample 8 and vote | | | |
| Claude Haiku, 3-shot | | | |

## Data
Spider 1.0 (Yu et al., 2018; CC BY-SA 4.0): 10k questions over 200 SQLite databases; dev databases never appear in
training. Choices are made on held-out training databases; dev is read once.

## Run it yourself
```bash
pip install torch && pip install -r requirements-dev.txt && pytest -q   # offline, no GPU
python scripts/download.py --fetch hf && python scripts/build_splits.py
python scripts/eval_model.py --backend hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --split val --limit 200
```

## Stack
Python · PyTorch · Transformers · PEFT · sqlglot · SQLite · llama.cpp · FastAPI · Gradio

## License
MIT (code). Spider is CC BY-SA 4.0; base models are Apache-2.0.
