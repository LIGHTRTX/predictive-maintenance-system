import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("database/machine_data.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        air_temp REAL,
        process_temp REAL,
        rpm REAL,
        torque REAL,
        current REAL,
        failure_prob REAL,
        overload_ratio REAL,
        rms REAL,
        risk_score REAL,
        status TEXT
    )
    """)
    conn.commit()
    return conn

def insert_log(conn, data):
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO logs (
        timestamp, air_temp, process_temp, rpm, torque,
        current, failure_prob, overload_ratio,
        rms, risk_score, status
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["air_temp"],
        data["process_temp"],
        data["rpm"],
        data["torque"],
        data["current"],
        data["failure_prob"],
        data["overload_ratio"],
        data["rms"],
        data["risk_score"],
        data["status"]
    ))
    conn.commit()