"""A miniature Spider release (same folder layout and JSON fields) built from code, for tests and smoke runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

TRAIN_DBS = ["concert_singer", "pets_1", "shop_1", "school_1", "library_1", "airline_1", "museum_2", "hotel_1"]
DEV_DBS = ["cars_1", "zoo_1"]
TEST_DBS = ["bank_1"]
CITIES = ["Paris", "Rome", "Oslo", "Lima", "Cairo", "Kyoto"]


def _make_db(path: Path, entity: str, rng: np.random.Generator, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {entity} (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER, city TEXT, score REAL)")
    conn.execute(f"CREATE TABLE {entity}_event (event_id INTEGER PRIMARY KEY, {entity}_id INTEGER, year INTEGER, amount REAL, "
                 f"FOREIGN KEY ({entity}_id) REFERENCES {entity}(id))")
    for i in range(1, n + 1):
        conn.execute(f"INSERT INTO {entity} VALUES (?, ?, ?, ?, ?)", (i, f"{entity[:3]}_{i:02d}", int(rng.integers(18, 70)), CITIES[int(rng.integers(len(CITIES)))],
                                                                      round(float(rng.uniform(0, 100)), 2)))
    for j in range(1, 2 * n + 1):
        conn.execute(f"INSERT INTO {entity}_event VALUES (?, ?, ?, ?)", (j, int(rng.integers(1, n + 1)), int(rng.integers(2000, 2020)), round(float(rng.uniform(1, 500)), 2)))
    conn.commit()
    conn.close()


def _questions(db_id: str, entity: str, rng: np.random.Generator) -> list[dict]:
    e, ev = entity, f"{entity}_event"
    age = int(rng.integers(25, 50))
    city = CITIES[int(rng.integers(len(CITIES)))]
    qa = [
        (f"How many {e} records are there?", f"SELECT count(*) FROM {e}"),
        (f"List the names of all {e} records.", f"SELECT name FROM {e}"),
        (f"What is the average age of {e} records?", f"SELECT avg(age) FROM {e}"),
        (f"Which {e} names are older than {age}?", f"SELECT name FROM {e} WHERE age > {age}"),
        (f"List names and ages of {e} records in {city} ordered by age.", f"SELECT name, age FROM {e} WHERE city = '{city}' ORDER BY age"),
        (f"How many {e} records are there in each city?", f"SELECT city, count(*) FROM {e} GROUP BY city"),
        (f"What is the name of the oldest {e}?", f"SELECT name FROM {e} ORDER BY age DESC LIMIT 1"),
        (f"What is the total event amount per {e} name?", f"SELECT T1.name, sum(T2.amount) FROM {e} AS T1 JOIN {ev} AS T2 ON T1.id = T2.{e}_id GROUP BY T1.name"),
        (f"Which {e} names have no events?", f"SELECT name FROM {e} WHERE id NOT IN (SELECT {e}_id FROM {ev})"),
        (f"Which cities have more than 4 {e} records?", f"SELECT city FROM {e} GROUP BY city HAVING count(*) > 4"),
        (f"List {e} names in {city} or older than 60.", f"SELECT name FROM {e} WHERE city = '{city}' OR age > 60"),
        (f"Which {e} names appear in both {city} and among those older than {age}?",
         f"SELECT name FROM {e} WHERE city = '{city}' INTERSECT SELECT name FROM {e} WHERE age > {age}"),
        (f"What is the maximum and minimum score of {e} records?", f"SELECT max(score), min(score) FROM {e}"),
        (f"Show events after 2015 for {e} records.", f"SELECT event_id FROM {ev} WHERE year > 2015"),
    ]
    return [{"db_id": db_id, "question": q, "query": sql, "query_toks": sql.split(), "question_toks": q.split(), "sql": {}} for q, sql in qa]


def make_spider(root: Path, seed: int = 0) -> Path:
    """Write <root>/spider_data/{database/, test_database/, *.json}; returns the spider_data folder."""
    rng = np.random.default_rng(seed)
    sd = Path(root) / "spider_data"
    tables = []
    split_qs = {"train_spider": [], "train_others": [], "dev": [], "test": []}
    for group, dbs in (("train", TRAIN_DBS), ("dev", DEV_DBS), ("test", TEST_DBS)):
        for db in dbs:
            entity = db.rsplit("_", 1)[0] if db[-1].isdigit() else db.split("_")[0]
            entity = entity.replace("_", "")
            folder = "test_database" if group == "test" else "database"
            _make_db(sd / folder / db / f"{db}.sqlite", entity, rng)
            qs = _questions(db, entity, rng)
            if group == "train":
                (split_qs["train_others"] if db == "hotel_1" else split_qs["train_spider"]).extend(qs)
            else:
                split_qs[group].extend(qs)
            tables.append({"db_id": db, "table_names_original": [entity, f"{entity}_event"], "table_names": [entity, f"{entity} event"],
                           "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "age"], [0, "city"], [0, "score"], [1, "event_id"],
                                                     [1, f"{entity}_id"], [1, "year"], [1, "amount"]],
                           "column_names": [], "column_types": [], "primary_keys": [1, 6], "foreign_keys": [[7, 1]]})
    sd.mkdir(parents=True, exist_ok=True)
    for name, qs in split_qs.items():
        (sd / f"{name}.json").write_text(json.dumps(qs, indent=1), encoding="utf-8")
    (sd / "tables.json").write_text(json.dumps(tables, indent=1), encoding="utf-8")
    return sd
