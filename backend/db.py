"""
ZeroLeakX — SQLite persistence.

Persists user behavioural baselines (so a user's personal model survives logout
and restart — the basis of the multi-user / two-person live demo) and a durable
audit log of security events (exportable for the SOC). Live session state stays
in memory; only what must outlive a process is written here.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "zeroleakx.db"
_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
              username TEXT PRIMARY KEY,
              baseline TEXT,        -- JSON: list of feature-vector windows
              pointer  TEXT,        -- JSON: list of [mean,std] pointer windows
              n_samples INTEGER DEFAULT 0,
              created REAL,
              updated REAL
            );
            CREATE TABLE IF NOT EXISTS audit(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL, t TEXT, session_id TEXT, username TEXT,
              level TEXT, msg TEXT
            );
            """
        )
        _conn.commit()
    return _conn


# ── Users / baselines ────────────────────────────────────────────────────────
def get_user(username: str) -> dict | None:
    row = conn().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        return None
    return {
        "username": row["username"],
        "baseline": json.loads(row["baseline"]) if row["baseline"] else [],
        "pointer": json.loads(row["pointer"]) if row["pointer"] else None,
        "n_samples": row["n_samples"],
    }


def upsert_user(username: str) -> None:
    now = time.time()
    conn().execute(
        "INSERT INTO users(username,created,updated) VALUES(?,?,?) "
        "ON CONFLICT(username) DO NOTHING",
        (username, now, now),
    )
    conn().commit()


def save_baseline(username: str, windows: list, pointer: list | None, n_samples: int) -> None:
    conn().execute(
        "INSERT INTO users(username,baseline,pointer,n_samples,created,updated) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(username) DO UPDATE SET "
        "baseline=excluded.baseline, pointer=excluded.pointer, "
        "n_samples=excluded.n_samples, updated=excluded.updated",
        (username, json.dumps(windows), json.dumps(pointer) if pointer else None,
         n_samples, time.time(), time.time()),
    )
    conn().commit()


def list_users() -> list[dict]:
    rows = conn().execute(
        "SELECT username, n_samples, updated FROM users ORDER BY updated DESC").fetchall()
    return [{"username": r["username"], "n_samples": r["n_samples"],
             "enrolled": (r["n_samples"] or 0) >= 3} for r in rows]


# ── Audit log ────────────────────────────────────────────────────────────────
def log(session_id: str, username: str, level: str, msg: str) -> None:
    conn().execute(
        "INSERT INTO audit(ts,t,session_id,username,level,msg) VALUES(?,?,?,?,?,?)",
        (time.time(), time.strftime("%H:%M:%S"), session_id, username, level, msg),
    )
    conn().commit()


def audit(limit: int = 200, username: str | None = None) -> list[dict]:
    if username:
        rows = conn().execute(
            "SELECT * FROM audit WHERE username=? ORDER BY id DESC LIMIT ?",
            (username, limit)).fetchall()
    else:
        rows = conn().execute(
            "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"t": r["t"], "session_id": r["session_id"], "username": r["username"],
             "level": r["level"], "msg": r["msg"]} for r in rows]
