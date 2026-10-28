"""Safe execution of model-written SQL, schema rendering for prompts, and result comparison.

Safety is layered, because the same executor runs untrusted model output during RL and behind the public demo:
  1. the database is opened read-only (`mode=ro` URI) with `PRAGMA query_only = ON`
  2. a static guard accepts exactly one SELECT / WITH … SELECT statement and rejects writes, PRAGMA, ATTACH
  3. a progress handler aborts any query that runs longer than the timeout (runaway joins, recursive CTEs)
  4. at most `max_rows` rows are fetched; truncation is reported
Result comparison follows Spider's execution-accuracy convention: row order matters only when the gold query has an
ORDER BY; columns may come in a different order; floats are compared after rounding.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

from .config import EXEC, PROMPT, ExecRules, PromptRules

FORBIDDEN = {"insert", "update", "delete", "drop", "create", "alter", "replace", "attach", "detach", "pragma", "vacuum", "reindex", "analyze", "begin",
             "commit", "rollback", "savepoint", "release", "truncate", "grant", "load_extension"}
_STRING_OR_IDENT = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None  # "guard: …", "timeout", "sqlite: …"
    truncated: bool = False
    ms: float = 0.0

    @property
    def empty(self) -> bool:
        return self.ok and not self.rows


# --- guard ------------------------------------------------------------------------------------------------------------
def strip_sql(sql: str) -> str:
    """Remove comments and surrounding whitespace / trailing semicolons."""
    s = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql or ""))
    return s.strip().rstrip(";").strip()


def guard(sql: str) -> str | None:
    """None if the statement may run; otherwise the reason it is refused."""
    s = strip_sql(sql)
    if not s:
        return "empty"
    masked = _STRING_OR_IDENT.sub("''", s)  # keywords inside literals / quoted identifiers do not count
    if ";" in masked:
        return "more than one statement"
    first = masked.split(None, 1)[0].lower()
    if first not in ("select", "with", "values"):
        return f"not a query (starts with {first!r})"
    words = set(re.findall(r"[a-z_]+", masked.lower()))
    bad = sorted(words & FORBIDDEN)
    if bad:
        return f"forbidden keyword {bad[0]!r}"
    return None


# --- execution --------------------------------------------------------------------------------------------------------
def connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")  # a few Spider databases have stray bytes
    conn.execute("PRAGMA query_only = ON")
    return conn


def execute(db_path: Path, sql: str, rules: ExecRules = EXEC, trusted: bool = False, conn: sqlite3.Connection | None = None) -> ExecResult:
    """Run one query. `trusted=True` skips the static guard (gold SQL); read-only mode and the timeout always apply."""
    if not trusted:
        reason = guard(sql)
        if reason:
            return ExecResult(False, error=f"guard: {reason}")
    own = conn is None
    conn = conn or connect_ro(db_path)
    deadline = time.perf_counter() + rules.timeout_s
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 20_000)
    t0 = time.perf_counter()
    try:
        cur = conn.execute(strip_sql(sql))
        rows = cur.fetchmany(rules.max_rows + 1)
        cols = [d[0] for d in (cur.description or [])]
        truncated = len(rows) > rules.max_rows
        return ExecResult(True, [tuple(r) for r in rows[: rules.max_rows]], cols, None, truncated, (time.perf_counter() - t0) * 1000)
    except sqlite3.OperationalError as e:
        msg = "timeout" if "interrupted" in str(e).lower() else f"sqlite: {e}"
        return ExecResult(False, error=msg, ms=(time.perf_counter() - t0) * 1000)
    except (sqlite3.Error, ValueError, OverflowError) as e:  # e.g. "You can only execute one statement at a time"
        return ExecResult(False, error=f"sqlite: {e}", ms=(time.perf_counter() - t0) * 1000)
    finally:
        conn.set_progress_handler(None, 0)
        if own:
            conn.close()


class Executor:
    """Keeps one read-only connection per database and caches gold results (they are executed many times in RL)."""

    def __init__(self, rules: ExecRules = EXEC) -> None:
        self.rules = rules
        self._conns: dict[str, sqlite3.Connection] = {}
        self._gold: dict[tuple[str, str], ExecResult] = {}

    def _conn(self, db_path: Path) -> sqlite3.Connection:
        key = str(Path(db_path).resolve())
        if key not in self._conns:
            self._conns[key] = connect_ro(Path(key))
        return self._conns[key]

    def run(self, db_path: Path, sql: str) -> ExecResult:
        return execute(db_path, sql, self.rules, trusted=False, conn=self._conn(db_path))

    def gold(self, db_path: Path, sql: str) -> ExecResult:
        key = (str(db_path), sql)
        if key not in self._gold:
            self._gold[key] = execute(db_path, sql, self.rules, trusted=True, conn=self._conn(db_path))
        return self._gold[key]

    def close(self) -> None:
        for c in self._conns.values():
            c.close()
        self._conns.clear()


# --- result comparison ------------------------------------------------------------------------------------------------
def _norm(v, digits: int):
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        r = round(v, digits)
        return int(r) if r.is_integer() else r
    if isinstance(v, bytes):
        return v.hex()
    return v


def _rows(res: ExecResult, digits: int) -> list[tuple]:
    return [tuple(_norm(v, digits) for v in r) for r in res.rows]


def has_order_by(sql: str) -> bool:
    masked = _STRING_OR_IDENT.sub("''", strip_sql(sql)).lower()
    return re.search(r"\border\s+by\b", masked) is not None


def results_match(pred: ExecResult, gold: ExecResult, ordered: bool, digits: int = EXEC.float_digits) -> bool:
    """Spider-style execution match: same rows (as a multiset, or as a list when `ordered`), allowing any permutation of
    columns. Both results must have executed."""
    if not (pred.ok and gold.ok):
        return False
    p, g = _rows(pred, digits), _rows(gold, digits)
    if len(p) != len(g):
        return False
    if not g:
        return len(pred.columns) == len(gold.columns) or not pred.columns or not gold.columns
    width = len(g[0])
    if any(len(r) != width for r in p):
        return False
    gold_cols = [sorted(map(repr, col)) for col in zip(*g)]
    pred_cols = [sorted(map(repr, col)) for col in zip(*p)]
    # candidate column mappings: a predicted column can only stand for a gold column with the same multiset of values
    options = [[j for j in range(width) if pred_cols[j] == gold_cols[i]] for i in range(width)]
    if any(not o for o in options):
        return False
    target = g if ordered else Counter(g)
    for perm in _assignments(options, width):
        q = [tuple(r[j] for j in perm) for r in p]
        if (q == target) if ordered else (Counter(q) == target):
            return True
    return False


def _assignments(options: list[list[int]], width: int, limit: int = 5040):
    """Injective choices of one predicted column per gold column (bounded; identity first)."""
    if all(i in o for i, o in enumerate(options)):
        yield tuple(range(width))
    count = 0
    for perm in permutations(range(width)):
        if all(perm[i] in options[i] for i in range(width)):
            yield perm
            count += 1
            if count >= limit:
                return


# --- schema rendering ---------------------------------------------------------------------------------------------------
def table_ddl(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """(table name, CREATE TABLE text) for every user table, in definition order."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall()
    out = []
    for name, sql in rows:
        if not sql:
            cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            sql = f'CREATE TABLE "{name}" (' + ", ".join(f'"{c[1]}" {c[2]}'.strip() for c in cols) + ")"
        out.append((name, re.sub(r"\s+", " ", sql).strip()))
    return out


def render_schema(db_path: Path, rules: PromptRules = PROMPT) -> str:
    """CREATE TABLE statements, each followed by up to `sample_rows` example rows as a comment block."""
    conn = connect_ro(db_path)
    try:
        parts = []
        for name, ddl in table_ddl(conn):
            block = ddl + ";"
            if rules.sample_rows > 0:
                try:
                    cur = conn.execute(f'SELECT * FROM "{name}" LIMIT {int(rules.sample_rows)}')
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                except sqlite3.Error:
                    rows, cols = [], []
                if rows:
                    cell = lambda v: (str(v)[: rules.max_cell_chars] + "…") if v is not None and len(str(v)) > rules.max_cell_chars else str(v)
                    lines = [" | ".join(cols)] + [" | ".join(cell(v) for v in r) for r in rows]
                    block += "\n/* " + f"{len(rows)} example rows:\n" + "\n".join(lines) + " */"
            parts.append(block)
        text, kept = "", 0
        for block in parts:
            if len(text) + len(block) + 2 > rules.max_schema_chars and kept > 0:
                text += f"\n/* {len(parts) - kept} more tables not shown */"
                break
            text += ("\n\n" if text else "") + block
            kept += 1
        return text
    finally:
        conn.close()
