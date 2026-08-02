import os
import sqlite3

class SqliteDatabase:
    def __init__(self, db_path: str = "/app/data/guard_data.db"):
        self.db_path = db_path

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row

        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    hashed_password TEXT NOT NULL,
                    is_admin BOOLEAN NOT NULL DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    objects_detected TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_analytics_event_id
                ON event_analytics(event_id)
            """)

            conn.commit()