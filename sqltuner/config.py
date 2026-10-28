"""Every knob: paths, splits, model candidates, prompt format, training defaults, prices. No secrets here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "eval" / "results"

SEED = 7

# Spider 1.0 (Yu et al., 2018; CC BY-SA 4.0). Two download routes, see scripts/download.py.
SPIDER_DIRNAME = "spider_data"
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"  # the official Yale release (spider_data.zip)
SPIDER_HF_MIRROR = "HAL-9001/spider-databases"  # re-host of the same zip, CC BY-SA 4.0
SPIDER_ZIP_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"  # as published by the mirror

# Open base models considered in M1 (all Apache-2.0). The chosen one is written to eval/results/base_choice.json.
BASE_CANDIDATES = (
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
)

# Hosted reference models (Anthropic API). Prices in USD per million tokens (input, output); check before quoting.
API_MODELS = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}


@dataclass(frozen=True)
class SplitRules:
    """Spider train → our train + val (val = whole held-out training databases, so it is cross-database like dev).
    Spider dev is the reported number and is read once (M4); Spider test likewise if the release contains it."""

    val_min_questions: int = 600
    val_max_dbs: int = 20


@dataclass(frozen=True)
class PromptRules:
    sample_rows: int = 3  # rows shown per table under its CREATE TABLE (0 = schema only)
    max_cell_chars: int = 40
    max_schema_chars: int = 12_000  # very large schemas are cut table by table, never mid-table
    few_shot: int = 0  # examples from other training databases (API baselines use 3)


@dataclass(frozen=True)
class ExecRules:
    timeout_s: float = 5.0
    max_rows: int = 10_000
    float_digits: int = 6


@dataclass(frozen=True)
class GenRules:
    max_new_tokens: int = 256
    temperature: float = 0.0  # greedy for evaluation
    vote_samples: int = 8  # self-consistency: sample N, execute, majority vote on results
    vote_temperature: float = 0.7


@dataclass(frozen=True)
class LoraRules:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class SFTRules:
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 4  # sequences per micro-batch
    grad_accum: int = 4
    max_len: int = 2048  # tokens; longer examples are dropped and counted
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    eval_every: int = 200  # optimizer steps between val-loss checks


@dataclass(frozen=True)
class GRPORules:
    steps: int = 300
    prompts_per_step: int = 4
    group_size: int = 8  # completions per prompt
    temperature: float = 1.0
    max_new_tokens: int = 192
    lr: float = 5e-6
    beta: float = 0.02  # KL to the reference (SFT) policy, k3 estimator
    clip_eps: float = 0.2
    inner_epochs: int = 1  # μ: optimisation passes over each batch of samples (1 = strictly on-policy)
    loss_agg: str = "token"  # "token" = mean over all completion tokens (DAPO); "sequence" = per-sequence mean (GRPO paper)
    micro_batch: int = 4  # sequences per forward/backward when computing log-probs
    exec_bonus: float = 0.1  # reward for SQL that runs but returns the wrong result
    skip_gold_empty: bool = True  # prompts whose gold result is empty are trivially matched by broken SQL
    eval_every: int = 50
    eval_questions: int = 200


SPLIT_RULES = SplitRules()
PROMPT = PromptRules()
EXEC = ExecRules()
GEN = GenRules()
LORA = LoraRules()
SFT = SFTRules()
GRPO = GRPORules()
BOOTSTRAP_RESAMPLES = 1000
HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")
SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema with a few example rows and a question, "
    "write one SQLite query that answers the question. Use only tables and columns from the schema. "
    "Reply with the query in a single ```sql code block and nothing else."
)
