"""The executor's safety layers, result comparison, schema rendering, SQL extraction, hardness."""

from __future__ import annotations

import pytest

from sqltuner.db import ExecResult, Executor, execute, guard, has_order_by, render_schema, results_match, strip_sql
from sqltuner.hardness import hardness
from sqltuner.prompts import extract_sql, normalize_sql


def db(spider):
    return spider["sd"] / "database" / "concert_singer" / "concert_singer.sqlite"


@pytest.mark.parametrize("sql,ok", [
    ("SELECT 1", True), ("with a as (select 1) select * from a", True), ("SELECT name FROM t WHERE name = 'drop table'", True),
    ("SELECT \"delete\" FROM t", True), ("DROP TABLE concert", False), ("SELECT 1; DROP TABLE x", False), ("PRAGMA table_info(x)", False),
    ("ATTACH DATABASE 'x' AS y", False), ("SELECT * FROM t; ", True), ("", False), ("-- comment only", False), ("SELECT load_extension('x')", False),
])
def test_guard(sql, ok):
    assert (guard(sql) is None) == ok, (sql, guard(sql))


def test_read_only_timeout_and_row_cap(spider):
    p = db(spider)
    assert "readonly" in execute(p, "INSERT INTO concert VALUES (99,'x',1,'y',1)", trusted=True).error  # even trusted SQL cannot write
    assert execute(p, "DELETE FROM concert").error.startswith("guard")
    r = execute(p, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c")
    assert r.error == "timeout" and r.ms < 8000
    from sqltuner.config import ExecRules

    r = execute(p, "SELECT * FROM concert_event", ExecRules(timeout_s=5, max_rows=10))
    assert r.ok and r.truncated and len(r.rows) == 10
    assert execute(p, "SELECT nope FROM concert").error.startswith("sqlite:")
    ex = Executor()
    g1 = ex.gold(p, "SELECT count(*) FROM concert")
    assert g1 is ex.gold(p, "SELECT count(*) FROM concert") and g1.rows == [(30,)]
    ex.close()


def test_row_cap_keeps_large_results_whole(spider):
    from sqltuner.config import EXEC

    big = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < {n}) SELECT x FROM c"
    r = execute(db(spider), big.format(n=15_000))
    assert EXEC.max_rows == 200_000 and r.ok and len(r.rows) == 15_000 and not r.truncated  # a 10,000 cap cut such results
    r = execute(db(spider), big.format(n=EXEC.max_rows + 5))
    assert r.ok and r.truncated and len(r.rows) == EXEC.max_rows


def test_results_match():
    a = ExecResult(True, [(1, "x"), (2, "y")], ["id", "name"])
    b = ExecResult(True, [("y", 2), ("x", 1)], ["name", "id"])
    assert results_match(b, a, ordered=False)  # column permutation + row order ignored
    assert not results_match(b, a, ordered=True)  # rows in the wrong order
    c = ExecResult(True, [("x", 1), ("y", 2)], ["name", "id"])
    assert results_match(c, a, ordered=True)
    assert results_match(ExecResult(True, [(0.30000000001,)], ["v"]), ExecResult(True, [(0.3,)], ["v"]), ordered=False)
    assert not results_match(ExecResult(True, [(1,), (1,)], ["v"]), ExecResult(True, [(1,)], ["v"]), ordered=False)  # multiset, not set
    assert not results_match(ExecResult(False, error="x"), a, ordered=False)
    assert results_match(ExecResult(True, [], ["a"]), ExecResult(True, [], ["b"]), ordered=False)
    assert has_order_by("select a from t ORDER  BY a") and not has_order_by("select 'order by' from t")


def test_schema_and_extraction(spider):
    s = render_schema(db(spider))
    assert "CREATE TABLE concert " in s and "FOREIGN KEY" in s and "3 example rows" in s
    from sqltuner.config import PromptRules

    assert "example rows" not in render_schema(db(spider), PromptRules(sample_rows=0))
    assert "more tables not shown" in render_schema(db(spider), PromptRules(max_schema_chars=200))
    assert extract_sql("```sql\nSELECT a\nFROM b;\n```\nThis query…") == "SELECT a FROM b"
    assert extract_sql("<think>long</think>\n```sql\nselect 1\n```") == "select 1"
    assert extract_sql("Sure! SELECT name FROM t\n\nExplanation: it selects") == "SELECT name FROM t"
    assert extract_sql("```sql\nSELECT x FROM y") == "SELECT x FROM y"  # unterminated block
    assert extract_sql("I cannot answer") == ""
    assert normalize_sql("SELECT  a ,  b\n FROM t ;") == "SELECT a , b FROM t"
    assert strip_sql("SELECT 1; -- x") == "SELECT 1"


@pytest.mark.parametrize("sql,level", [
    ("SELECT count(*) FROM singer", "easy"),
    ("SELECT name FROM singer WHERE age > 30", "easy"),
    ("SELECT name, age FROM singer WHERE age > 30 ORDER BY age", "medium"),
    ("SELECT T1.name, count(*) FROM a AS T1 JOIN b AS T2 ON T1.id = T2.aid GROUP BY T1.name", "medium"),
    ("SELECT name FROM singer WHERE name LIKE '%a%'", "medium"),  # a lone LIKE counts (official evaluation.py)
    ("SELECT min(age), max(age) FROM singer WHERE a = 1 OR b = 2", "hard"),  # aggregates straight in SELECT count
    ("SELECT name FROM singer WHERE id NOT IN (SELECT singer_id FROM concert)", "hard"),
    ("SELECT name FROM a WHERE x = 1 INTERSECT SELECT name FROM a WHERE y = 2", "hard"),
    ("SELECT T1.name FROM a AS T1 JOIN b AS T2 ON T1.id = T2.aid JOIN c AS T3 ON T2.id = T3.bid WHERE T3.v > (SELECT avg(v) FROM c) GROUP BY T1.name ORDER BY count(*) DESC LIMIT 1", "extra"),
    ("this is not sql", "unknown"),
])
def test_hardness(sql, level):
    assert hardness(sql) == level
