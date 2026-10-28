"""Chat messages for a question, the SFT target, and robust extraction of the SQL from a model's reply."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import PROMPT, SYSTEM_PROMPT, PromptRules
from .db import render_schema

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*)$", re.DOTALL | re.IGNORECASE)
_STATEMENT = re.compile(r"\b(with|select)\b.*", re.DOTALL | re.IGNORECASE)


@lru_cache(maxsize=512)
def schema_text(db_path: str, sample_rows: int = PROMPT.sample_rows, max_cell_chars: int = PROMPT.max_cell_chars, max_schema_chars: int = PROMPT.max_schema_chars) -> str:
    return render_schema(Path(db_path), PromptRules(sample_rows, max_cell_chars, max_schema_chars))


def user_message(schema: str, question: str) -> str:
    return f"Database schema:\n{schema}\n\nQuestion: {question}"


def target(sql: str) -> str:
    return f"```sql\n{normalize_sql(sql)}\n```"


def normalize_sql(sql: str) -> str:
    """Collapse whitespace (Spider gold queries carry odd spacing); keep the text otherwise identical."""
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").strip()


def messages(schema: str, question: str, shots: list[tuple[str, str, str]] = (), system: str = SYSTEM_PROMPT) -> list[dict]:
    """shots: (schema, question, sql) from OTHER databases — never the question's own database."""
    msgs = [{"role": "system", "content": system}]
    for s_schema, s_q, s_sql in shots:
        msgs.append({"role": "user", "content": user_message(s_schema, s_q)})
        msgs.append({"role": "assistant", "content": target(s_sql)})
    msgs.append({"role": "user", "content": user_message(schema, question)})
    return msgs


def extract_sql(reply: str) -> str:
    """The first ```sql block; else an unterminated block; else the text from the first SELECT/WITH. Thinking tags removed."""
    text = re.sub(r"<think>.*?</think>", " ", reply or "", flags=re.DOTALL | re.IGNORECASE)
    m = _FENCE.search(text) or _OPEN_FENCE.search(text)
    body = m.group(1) if m else text
    s = _STATEMENT.search(body)
    if not s:
        return ""
    sql = s.group(0)
    sql = sql.split("```")[0]
    sql = re.split(r"\n\s*\n", sql.strip())[0]  # stop at the first blank line (explanations after the query)
    return normalize_sql(sql)


def apply_chat_template(tokenizer, msgs: list[dict], add_generation_prompt: bool = True) -> str:
    """Tokenizer's own chat template; Qwen3 models get thinking switched off (the task wants the SQL, not a trace)."""
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
