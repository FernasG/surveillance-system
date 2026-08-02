import asyncio, json, logging
from redis import asyncio as aioredis

from guard.pipeline.description.description_service import DescriptionService
from guard.infrastructure.messaging.search_priority_listener import SearchPriorityListener

logger = logging.getLogger(__name__)

class RedisEventQueueWorker:
    def __init__(
        self,
        redis_client: aioredis.Redis,
        description_service: DescriptionService,
        priority_listener: SearchPriorityListener,
    ):
        self.description_service = description_service
        self.priority_listener = priority_listener
        self._redis_client = redis_client

        self.priority_listener.set_on_started_callback(self.description_service.cancel_in_flight)

    async def start(self, queue_name: str):
        logger.info("Starting Description Worker (Async)")

        try:
            while True:
                try:
                    result = await self._redis_client.brpop(queue_name)

                    if not result:
                        continue

                    _, message = result
                    event = json.loads(message)

                    if self.priority_listener.search_active.is_set():
                        logger.info("Search in progress, deferring description generation")
                        await self.priority_listener.not_searching.wait()

                    outcome = await self.description_service.process(
                        event_id=event["event_id"],
                        video_path=event["video_path"],
                        elapsed_ms=event["elapsed_ms"],
                    )

                    if outcome is None:
                        await self._redis_client.lpush(queue_name, json.dumps(event))

                except json.JSONDecodeError:
                    logger.error("Failed to decode Redis message JSON.")
                except Exception as e:
                    logger.error(f"Unexpected error in description worker pipeline: {str(e)}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Description worker loop cancelled, shutting down")
        finally:
            await self.stop()

    async def stop(self):
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed successfully.")

        logger.info("Description Worker fully stopped.")
