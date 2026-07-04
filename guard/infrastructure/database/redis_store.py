import redis
import redis.asyncio as aioredis
from typing import Optional
from guard.core.entities import User
from guard.core.interfaces import IAuthRepository

class RedisStore(IAuthRepository):
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def get_user(self, username: str) -> Optional[User]:
        data = await self.redis.hgetall(f"user:{username}")

        if not data:
            return None
        
        return User(
            username=username,
            hashed_password=data[b"hashed_password"].decode("utf-8"),
            is_admin=(data[b"is_admin"].decode("utf-8") == "True")
        )

    async def save_user(self, user: User) -> None:
        await self.redis.hset(
            f"user:{user.username}", 
            mapping={
                "hashed_password": user.hashed_password,
                "is_admin": str(user.is_admin)
            }
        )
