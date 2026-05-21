"""
SQLite storage for the local OMR app.

The current app intentionally avoids login, cloud accounts, and a server
database. SQLite gives us a single local file that is easy to inspect, copy, or
back up for one teacher or one computer.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "omr_demo.sqlite3"


def ensure_storage_dirs() -> None:
    """Create local folders used by the webapp if they do not already exist."""

    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection configured for dictionary-like rows.

    Each request opens a short-lived connection. That is simple and reliable for
    a local app where one teacher is using the system at a time.
    """

    ensure_storage_dirs()
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """
    Create the database tables if they are missing.

    The schema separates a test from its variations. A test stores the common
    information such as name and question count. Each variation stores its own
    answer key, so Version A and Version B can have different correct answers.
    """

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                question_count INTEGER NOT NULL,
                options_json TEXT NOT NULL,
                questions_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS test_variations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                answer_key_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                variation_id INTEGER NOT NULL,
                student_name TEXT,
                student_id TEXT,
                image_path TEXT,
                annotated_image_path TEXT,
                detected_answers_json TEXT NOT NULL,
                detailed_results_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE,
                FOREIGN KEY (variation_id) REFERENCES test_variations(id) ON DELETE CASCADE
            );
            """
        )
        _ensure_column(
            connection,
            table_name="tests",
            column_name="questions_json",
            column_definition="questions_json TEXT NOT NULL DEFAULT '{}'",
        )


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """
    Add a missing column to an existing local database.

    The app is being built iteratively. If an older SQLite file already exists
    on the teacher's computer, CREATE TABLE IF NOT EXISTS will not modify it.
    This tiny migration helper keeps the database compatible without requiring
    the teacher to delete their saved results.
    """

    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }

    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")


def create_variation(test_id: int, name: str, answer_key: dict[str, str]) -> int:
    """Insert one answer-key variation for an existing test."""

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO test_variations (test_id, name, answer_key_json)
            VALUES (?, ?, ?)
            """,
            (test_id, name, json.dumps(answer_key)),
        )

        return int(cursor.lastrowid)


def create_test_with_variation(
    name: str,
    question_count: int,
    options: list[str],
    question_texts: dict[str, str],
    variation_name: str,
    answer_key: dict[str, str],
) -> tuple[int, int]:
    """Create a test and its first variation in one transaction."""

    with get_connection() as connection:
        test_cursor = connection.execute(
            """
            INSERT INTO tests (name, question_count, options_json, questions_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                question_count,
                json.dumps(options),
                json.dumps(question_texts),
            ),
        )
        test_id = int(test_cursor.lastrowid)

        variation_cursor = connection.execute(
            """
            INSERT INTO test_variations (test_id, name, answer_key_json)
            VALUES (?, ?, ?)
            """,
            (test_id, variation_name, json.dumps(answer_key)),
        )
        variation_id = int(variation_cursor.lastrowid)

    return test_id, variation_id


def list_tests() -> list[sqlite3.Row]:
    """Return tests with a count of saved variations."""

    with get_connection() as connection:
        return connection.execute(
            """
            SELECT
                tests.*,
                COUNT(test_variations.id) AS variation_count
            FROM tests
            LEFT JOIN test_variations ON test_variations.test_id = tests.id
            GROUP BY tests.id
            ORDER BY tests.created_at DESC, tests.id DESC
            """
        ).fetchall()


def get_test(test_id: int) -> sqlite3.Row | None:
    """Return one test row by ID."""

    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM tests WHERE id = ?",
            (test_id,),
        ).fetchone()


def list_variations(test_id: int) -> list[sqlite3.Row]:
    """Return all variations for one test."""

    with get_connection() as connection:
        return connection.execute(
            """
            SELECT * FROM test_variations
            WHERE test_id = ?
            ORDER BY name COLLATE NOCASE
            """,
            (test_id,),
        ).fetchall()


def get_variation(variation_id: int) -> sqlite3.Row | None:
    """Return one variation row by ID."""

    with get_connection() as connection:
        return connection.execute(
            """
            SELECT
                test_variations.*,
                tests.name AS test_name,
                tests.question_count,
                tests.options_json
            FROM test_variations
            JOIN tests ON tests.id = test_variations.test_id
            WHERE test_variations.id = ?
            """,
            (variation_id,),
        ).fetchone()


def save_scan_result(
    test_id: int,
    variation_id: int,
    student_name: str,
    student_id: str,
    image_path: str | None,
    annotated_image_path: str | None,
    detected_answers: dict,
    detailed_results: dict,
    summary: dict,
) -> int:
    """Persist a confirmed scan result and return its ID."""

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO scan_results (
                test_id,
                variation_id,
                student_name,
                student_id,
                image_path,
                annotated_image_path,
                detected_answers_json,
                detailed_results_json,
                summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test_id,
                variation_id,
                student_name,
                student_id,
                image_path,
                annotated_image_path,
                json.dumps(detected_answers),
                json.dumps(detailed_results),
                json.dumps(summary),
            ),
        )

        return int(cursor.lastrowid)


def list_results() -> list[dict]:
    """
    Return saved scan results with decoded JSON fields.

    Decoding JSON in this layer keeps the Flask templates simple and avoids
    repeating json.loads in several route functions.
    """

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                scan_results.*,
                tests.name AS test_name,
                test_variations.name AS variation_name
            FROM scan_results
            JOIN tests ON tests.id = scan_results.test_id
            JOIN test_variations ON test_variations.id = scan_results.variation_id
            ORDER BY scan_results.created_at DESC, scan_results.id DESC
            """
        ).fetchall()

    results = []

    for row in rows:
        result = dict(row)
        result["summary"] = json.loads(result.pop("summary_json"))
        result["detected_answers"] = json.loads(result.pop("detected_answers_json"))
        result["detailed_results"] = json.loads(result.pop("detailed_results_json"))
        results.append(result)

    return results
