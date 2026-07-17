import asyncio
from typing import Optional
from guard.core.entities import User
from guard.core.interfaces import IAuthRepository
from guard.infrastructure.database.sqlite_store import SqliteDatabase

class SqliteAuthStore(IAuthRepository):
    def __init__(self, db: SqliteDatabase):
        self.db = db

    async def get_user(self, username: str) -> Optional[User]:
        return await asyncio.to_thread(self._get_user_sync, username)

    async def save_user(self, user: User) -> None:
        await asyncio.to_thread(self._save_user_sync, user)
    
    def _get_user_sync(self, username: str) -> Optional[User]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT username, hashed_password, is_admin FROM users WHERE username = ?", 
                (username,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return User(
                username=row["username"],
                hashed_password=row["hashed_password"],
                is_admin=bool(row["is_admin"])
            )

    def _save_user_sync(self, user: User) -> None:
        with self.db.get_connection() as conn:
            
            conn.execute(
                """
                INSERT INTO users (username, hashed_password, is_admin)
                VALUES (?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    hashed_password = excluded.hashed_password,
                    is_admin = excluded.is_admin
                """,
                (user.username, user.hashed_password, int(user.is_admin))
            )

            conn.commit()