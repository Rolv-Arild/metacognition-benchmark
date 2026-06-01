"""SQLite database for caching Wikipedia articles and generated questions."""

import json
import sqlite3
from pathlib import Path
from typing import Optional


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            language TEXT NOT NULL,
            text TEXT NOT NULL,
            pageviews INTEGER,
            difficulty INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answers TEXT NOT NULL,
            category TEXT NOT NULL,
            source_entity TEXT NOT NULL,
            source_label TEXT NOT NULL,
            difficulty INTEGER NOT NULL,
            salience INTEGER NOT NULL DEFAULT 3,
            global_relevance INTEGER NOT NULL DEFAULT 3,
            UNIQUE(source_entity, question)
        )
    """)
    conn.commit()
    return conn


def upsert_article(conn: sqlite3.Connection, title: str, language: str,
                    text: str, url: str, pageviews: Optional[int], difficulty: int):
    """Insert or update an article."""
    conn.execute(
        "INSERT OR REPLACE INTO articles (url, title, language, text, pageviews, difficulty) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (url, title, language, text, pageviews, difficulty),
    )
    conn.commit()


def load_articles(conn: sqlite3.Connection) -> list[dict]:
    """Load all cached articles."""
    rows = conn.execute("SELECT * FROM articles").fetchall()
    return [dict(row) for row in rows]


def count_articles_by_difficulty(conn: sqlite3.Connection) -> dict[int, int]:
    """Count articles per difficulty level."""
    rows = conn.execute(
        "SELECT difficulty, COUNT(*) as cnt FROM articles GROUP BY difficulty"
    ).fetchall()
    return {row["difficulty"]: row["cnt"] for row in rows}


def insert_question(conn: sqlite3.Connection, question: str, answers: list[str],
                    category: str, source_entity: str, source_label: str,
                    difficulty: int, salience: int = 3, global_relevance: int = 3):
    """Insert a question (ignores duplicates)."""
    conn.execute(
        "INSERT OR IGNORE INTO questions "
        "(question, answers, category, source_entity, source_label, difficulty, salience, global_relevance) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (question, json.dumps(answers, ensure_ascii=False), category,
         source_entity, source_label, difficulty, salience, global_relevance),
    )
    conn.commit()


def load_questions_db(conn: sqlite3.Connection) -> list[dict]:
    """Load all generated questions."""
    rows = conn.execute("SELECT * FROM questions").fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["answers"] = json.loads(d["answers"])
        results.append(d)
    return results


def article_exists(conn: sqlite3.Connection, url: str) -> bool:
    """Check if an article is already cached."""
    row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
    return row is not None

