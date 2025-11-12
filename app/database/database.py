from pathlib import Path
import logging
import sqlite3
import os

# --- Shared database location ---
DB_FILENAME = "environment.db"
BASE_DIR = Path(__file__).resolve().parent
LEGACY_PATH = BASE_DIR.parent.parent / DB_FILENAME


# Determine canonical DB path and migrate legacy copies when needed.
def get_database_path() -> str:
    """
    Return the path to the SQLite database located under app/database.

    If a legacy database exists in the project root, attempt to move it into the
    canonical location so the application keeps working without manual steps.
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    canonical_path = BASE_DIR / DB_FILENAME

    # Move the legacy database into place if the new location is still empty.
    if LEGACY_PATH.exists() and not canonical_path.exists():
        try:
            LEGACY_PATH.replace(canonical_path)
            logging.info("Moved legacy database from %s to %s.", LEGACY_PATH, canonical_path)
        except OSError as exc:
            logging.warning(
                "Unable to move legacy database from %s to %s: %s",
                LEGACY_PATH,
                canonical_path,
                exc,
            )

    return str(canonical_path)


db_file = get_database_path()


# --- Database connection helper ---
# Open a SQLite connection with WAL enabled for concurrency.
def get_connection():
    conn = sqlite3.connect(db_file, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enables concurrent access
    return conn


# --- Database functions ---
# Create the environment table if it does not yet exist.
def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS environment(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            illumination REAL,
            motion INTEGER,
            co2 REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {db_file}")


# Insert a new row of sensor data into the requested table.
def database_insert(table, timestamp, temp, light, motion, co2):
    motion_val = int(motion) if motion is not None else 0
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT INTO {table} (timestamp, temperature, illumination, motion, co2)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, temp, light, motion_val, co2))
    conn.commit()
    conn.close()
    print(f"🟢 Data inserted into table '{table}' at {timestamp}.")


# Fetch the most recent record for the given table.
def database_get_latest(table):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT * FROM {table}
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()
    # Only return a result when the query produced a row.
    if row:
        return {
            "id": row[0],
            "timestamp": row[1],
            "temperature": row[2],
            "illumination": row[3],
            "motion": bool(row[4]),
            "co2": row[5]
        }
    return None


# Fetch the latest `limit` rows for historical charts/lists.
def database_get_recent(table, limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT * FROM {table}
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    # Build a list of dicts so the API can serialize the rows cleanly.
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "temperature": row[2],
            "illumination": row[3],
            "motion": bool(row[4]),
            "co2": row[5]
        }
        for row in rows
    ]