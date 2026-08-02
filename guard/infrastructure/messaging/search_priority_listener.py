import asyncio
from redis import asyncio as aioredis
from loguru import logger

class SearchPriorityListener:
    def __init__(self, redis_client: aioredis.Redis, channel: str):
        self._redis_client = redis_client
        self._channel = channel
        self.search_active = asyncio.Event()

        self.not_searching = asyncio.Event()
        self.not_searching.set()

        self._on_started_callback = None

    def set_on_started_callback(self, callback) -> None:
        self._on_started_callback = callback

    async def start(self) -> None:
        pubsub = self._redis_client.pubsub()
        await pubsub.subscribe(self._channel)

        logger.info(f"Subscribed to search priority channel: {self._channel}")

        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                data = message["data"]

                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                if data == "started":
                    self.search_active.set()
                    self.not_searching.clear()

                    if self._on_started_callback:
                        self._on_started_callback()
                elif data == "finished":
                    self.search_active.clear()
                    self.not_searching.set()
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.close()
