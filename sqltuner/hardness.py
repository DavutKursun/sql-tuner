"""Spider's difficulty levels (easy / medium / hard / extra) for a gold query, re-implemented on the sqlglot AST.

The official rule (taoyds/spider, evaluation.py `eval_hardness`) counts three things on the top-level query:
  component1: WHERE, GROUP BY, ORDER BY, LIMIT (one each if present), extra tables in FROM (joins),
              OR connectives and LIKE conditions in ON / WHERE / HAVING
  component2: nested queries — subqueries inside WHERE / HAVING plus UNION / INTERSECT / EXCEPT
  others:     more than one aggregate, more than one selected column, more than one WHERE condition,
              more than one GROUP BY column (one point each)
and maps them to a level. This port is checked against the official script in Milestone 1 (agreement reported); if it
disagrees on more than 10% of training questions, the official script is vendored instead.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def _conds(node: exp.Expression | None) -> list[exp.Expression]:
    """Flatten an AND/OR tree into its condition leaves."""
    if node is None:
        return []
    if isinstance(node, (exp.And, exp.Or)):
        return _conds(node.left) + _conds(node.right)
    if isinstance(node, exp.Paren):
        return _conds(node.this)
    return [node]


def _count_or(node: exp.Expression | None) -> int:
    return 0 if node is None else len(list(node.find_all(exp.Or)))


def _shallow(node: exp.Expression | None, kind) -> list[exp.Expression]:
    """Nodes of `kind` under `node` (or `node` itself) that are not inside a nested SELECT."""
    if node is None:
        return []
    out = []
    for n in node.find_all(kind):
        p = n
        while p is not node and not isinstance(p, exp.Select):  # climb until `node` (shallow) or a SELECT (nested)
            p = p.parent
        if p is node:
            out.append(n)
    return out


def counts(sql: str) -> tuple[int, int, int]:
    tree = sqlglot.parse_one(sql, read="sqlite")
    comp2 = 0
    main = tree
    while isinstance(main, (exp.Union, exp.Intersect, exp.Except)):
        comp2 += 1
        main = main.left
    if isinstance(main, exp.Subquery):
        main = main.this
    if not isinstance(main, exp.Select):
        raise ValueError("not a query")
    where = main.args.get("where")
    having = main.args.get("having")
    group = main.args.get("group")
    order = main.args.get("order")
    joins = main.args.get("joins") or []
    where_node = where.this if where is not None else None
    having_node = having.this if having is not None else None
    on_nodes = [j.args.get("on") for j in joins if j.args.get("on") is not None]

    comp1 = sum(x is not None for x in (where, group, order)) + (1 if main.args.get("limit") is not None else 0)
    comp1 += len(joins)
    for n in [where_node, having_node, *on_nodes]:
        comp1 += _count_or(n)
        comp1 += len(_shallow(n, exp.Like)) if n is not None else 0
    for n in (where_node, having_node):
        if n is not None:
            comp2 += len([s for s in n.find_all(exp.Select) if s.parent_select is main or s.find_ancestor(exp.Select) is main])

    aggs = 0
    for part in [*main.expressions, where_node, having_node, group, order]:
        aggs += len(_shallow(part, exp.AggFunc)) if part is not None else 0
    others = int(aggs > 1) + int(len(main.expressions) > 1) + int(len(_conds(where_node)) > 1)
    others += int(group is not None and len(group.expressions) > 1)
    return comp1, comp2, others


def hardness(sql: str) -> str:
    try:
        c1, c2, o = counts(sql)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError, AttributeError):
        return "unknown"
    if c1 <= 1 and o == 0 and c2 == 0:
        return "easy"
    if (o <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and o < 2 and c2 == 0):
        return "medium"
    if (o > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and o <= 2 and c2 == 0) or (c1 <= 1 and o == 0 and c2 <= 1):
        return "hard"
    return "extra"
