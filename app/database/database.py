import sqlite3
import os
import platform

# --- Shared database location ---
def get_database_path():
    """
    Returns a path that works on both Windows and WSL.
    The database is stored in your Windows home directory,
    so both environments can read/write to it safely.
    """
    # Use the Windows home directory (accessible from both systems)
    win_path = r"C:\Users\Felix\Test\environment.db"
    wsl_path = "/mnt/c/Users/Felix/Test/environment.db"

    # Detect environment
    if "microsoft" in platform.uname().release.lower():
        # 🐧 Running inside WSL
        path = wsl_path
    else:
        # 🪟 Running in Windows
        path = win_path

    # Make sure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return os.path.abspath(path)


db_file = get_database_path()


# --- Database connection helper ---
def get_connection():
    conn = sqlite3.connect(db_file, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enables concurrent access
    return conn


# --- Database functions ---
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
    print(f"Database initialized at: {db_file}")


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
    print(f"Data inserted into table '{table}' at {timestamp}.")


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
    if row:
        return {
            "id": row[0],
            "timestamp": row[1],
            "temperature": row[2],
            "illumination": row[3],
            "motion": bool(row[4]),
            "co2": row[5]
        }
    else:
        return None


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
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "temperature": row[2],
            "illumination": row[3],
            "motion": bool(row[4]),
            "co2": row[5]
        } for row in rows
    ]
