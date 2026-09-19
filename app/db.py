"""Thin SQLite wrapper. Named analytics queries live in sql/analytics.sql."""
import re
import sqlite3
import threading
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


class Database:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript((SQL_DIR / "schema.sql").read_text())
        self._queries = self._load_queries()

    @staticmethod
    def _load_queries() -> dict:
        text = (SQL_DIR / "analytics.sql").read_text()
        parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
        return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}

    def execute(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executemany(self, sql, rows):
        with self.lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql, params=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def named(self, name: str, **params) -> list[dict]:
        return self.query(self._queries[name], params)
