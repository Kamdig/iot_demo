import sqlite3

db_file = 'environment.db'

def initialize_database():
    conn = sqlite3.connect(db_file)
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
    print("Database initialized.")

def database_insert(table, timestamp, temp, light, motion, co2):
    motion_val = int(motion) if motion is not None else 0
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f'''
                INSERT INTO {table} (timestamp, temperature, illumination, motion, co2)
                VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, temp, light, motion_val, co2))
    conn.commit()
    conn.close()
    print(f"Data inserted into table '{table}' at {timestamp}.")

def database_get_latest(table):
    conn = sqlite3.connect(db_file)
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
    conn = sqlite3.connect(db_file)
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