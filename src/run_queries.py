"""Run a SQL file from the sql/ folder and print each query's result.

Usage (from the project root):
    python -m src.run_queries 03_business_queries.sql
"""
import argparse
import re
import sqlite3
import sys
from contextlib import closing

import pandas as pd

from src.config import DB_PATH, SQL_DIR

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split a SQL file into (title, statement) pairs. Titles come from '-- Q<n>: ...' lines."""
    queries = []
    for chunk in sql_text.split(";"):
        lines = [line for line in chunk.strip().splitlines()]
        has_code = any(line.strip() and not line.strip().startswith("--") for line in lines)
        if not has_code:
            continue
        match = re.search(r"^--\s*(Q\d+.*)$", chunk, flags=re.MULTILINE)
        title = match.group(1) if match else "Untitled query"
        queries.append((title, chunk.strip()))
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a SQL file against the churn database.")
    parser.add_argument("sql_file", help="File name inside the sql/ folder, e.g. 03_business_queries.sql")
    args = parser.parse_args()

    sql_path = SQL_DIR / args.sql_file
    if not sql_path.exists():
        print(f"SQL file not found: {sql_path}")
        return 1
    if not DB_PATH.exists():
        print("Database not found. Run first: python -m src.db_setup")
        return 1

    with closing(sqlite3.connect(DB_PATH)) as conn:
        for title, statement in split_queries(sql_path.read_text(encoding="utf-8")):
            print(f"\n=== {title} ===")
            try:
                result = pd.read_sql_query(statement, conn)
                print(result.to_string(index=False))
            except (sqlite3.Error, pd.errors.DatabaseError) as error:
                print(f"Query failed: {error}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())