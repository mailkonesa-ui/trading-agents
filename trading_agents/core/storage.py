"""
SQLite-backed storage for proposals, decisions, and realised P&L.
Keeps it small and dependency-free.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trading.db"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                agent TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                price REAL NOT NULL,
                decision TEXT,             -- NULL | 'ACCEPT' | 'REJECT' | 'EXPIRED'
                decided_at TEXT,
                realised_pnl REAL          -- filled in later when next proposal of same ticker arrives
            );

            CREATE TABLE IF NOT EXISTS agent_stats (
                agent TEXT PRIMARY KEY,
                weight REAL NOT NULL DEFAULT 1.0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT
            );

            CREATE TABLE IF NOT EXISTS params (
                agent TEXT PRIMARY KEY,
                params_json TEXT NOT NULL
            );
            """
        )


def insert_proposal(
    ticker: str,
    agent: str,
    action: str,
    confidence: float,
    reason: str,
    price: float,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO proposals(created_at,ticker,agent,action,confidence,reason,price) "
            "VALUES(?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), ticker, agent, action, confidence, reason, price),
        )
        return int(cur.lastrowid)


def decide_proposal(pid: int, decision: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE proposals SET decision=?, decided_at=? WHERE id=?",
            (decision, datetime.utcnow().isoformat(), pid),
        )


def get_agent_stats(agent: str) -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM agent_stats WHERE agent=?", (agent,)
        ).fetchone()
        if row is None:
            return {"agent": agent, "weight": 1.0, "wins": 0, "losses": 0}
        return dict(row)


def upsert_agent_stats(agent: str, weight: float, wins: int, losses: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO agent_stats(agent,weight,wins,losses,last_updated) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(agent) DO UPDATE SET weight=excluded.weight, "
            "wins=excluded.wins, losses=excluded.losses, last_updated=excluded.last_updated",
            (agent, weight, wins, losses, datetime.utcnow().isoformat()),
        )


def all_agent_stats() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM agent_stats").fetchall()
        return [dict(r) for r in rows]


def recent_proposals(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM proposals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def open_accepted_proposal_for(ticker: str) -> dict | None:
    """Return the most recent accepted BUY/SELL for a ticker that has no P&L yet."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM proposals WHERE ticker=? AND decision='ACCEPT' "
            "AND realised_pnl IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        return dict(row) if row else None


def close_open_proposal(pid: int, pnl: float) -> None:
    with _conn() as c:
        c.execute("UPDATE proposals SET realised_pnl=? WHERE id=?", (pnl, pid))


def save_params(agent: str, params: dict) -> None:
    import json
    with _conn() as c:
        c.execute(
            "INSERT INTO params(agent,params_json) VALUES(?,?) "
            "ON CONFLICT(agent) DO UPDATE SET params_json=excluded.params_json",
            (agent, json.dumps(params)),
        )


def load_params(agent: str) -> dict | None:
    import json
    with _conn() as c:
        row = c.execute("SELECT params_json FROM params WHERE agent=?", (agent,)).fetchone()
        if row is None:
            return None
        return json.loads(row["params_json"])
