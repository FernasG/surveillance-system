import asyncio, json, logging
from redis import asyncio as aioredis

from guard.core.entities import SEARCH_GATE_KEY
from guard.core.exceptions import VLMGenerationError
from guard.pipeline.description.description_service import DescriptionService
from guard.infrastructure.messaging.search_priority_listener import SearchPriorityListener

logger = logging.getLogger(__name__)

class RedisEventQueueWorker:
    MAX_VLM_ATTEMPTS = 5
    VLM_RETRY_BACKOFF_S = 5
    SEARCH_WAIT_RECHECK_S = 15

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

        processing_queue = f"{queue_name}:processing"

        try:
            await self._reclaim_pending(queue_name, processing_queue)

            while True:
                message = await self._redis_client.blmove(queue_name, processing_queue, 0, "RIGHT", "LEFT")

                if not message:
                    continue

                try:
                    event = json.loads(message)

                    if self.priority_listener.search_active.is_set():
                        logger.info("Search in progress, deferring description generation")
                        await self._wait_until_no_search()

                    await self._handle_event(queue_name, event)
                except json.JSONDecodeError:
                    logger.error("Failed to decode Redis message JSON.")
                except Exception as e:
                    logger.error(f"Unexpected error in description worker pipeline: {str(e)}", exc_info=True)

                await self._ack(processing_queue, message)

        except asyncio.CancelledError:
            logger.info("Description worker loop cancelled, shutting down")
        finally:
            await self.stop()

    async def _handle_event(self, queue_name: str, event: dict) -> None:
        try:
            outcome = await self.description_service.process(
                event_id=event["event_id"],
                video_path=event["video_path"],
                elapsed_ms=event["elapsed_ms"],
            )
        except VLMGenerationError as e:
            attempts = event.get("vlm_attempts", 0) + 1

            if attempts >= self.MAX_VLM_ATTEMPTS:
                logger.error(f"Dropping event {event.get('event_id')} after {attempts} failed VLM attempts: {e}")
                return

            logger.warning(f"VLM generation failed for event {event.get('event_id')} (attempt {attempts}), re-queueing: {e}")
            event["vlm_attempts"] = attempts
            await self._redis_client.lpush(queue_name, json.dumps(event))

            await asyncio.sleep(self.VLM_RETRY_BACKOFF_S)
            return

        if outcome is None:
            await self._redis_client.lpush(queue_name, json.dumps(event))

    async def _wait_until_no_search(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self.priority_listener.not_searching.wait(),
                    timeout=self.SEARCH_WAIT_RECHECK_S,
                )
            except asyncio.TimeoutError:
                pass

            raw = await self._redis_client.get(SEARCH_GATE_KEY)
            active_searches = int(raw) if raw else 0

            if active_searches <= 0:
                self.priority_listener.search_active.clear()
                self.priority_listener.not_searching.set()
                return

    async def _reclaim_pending(self, queue_name: str, processing_queue: str) -> None:
        reclaimed = 0

        while await self._redis_client.lmove(processing_queue, queue_name, "RIGHT", "RIGHT"):
            reclaimed += 1

        if reclaimed:
            logger.info(f"Reclaimed {reclaimed} in-flight event(s) from a previous run")

    async def _ack(self, processing_queue: str, message) -> None:
        await self._redis_client.lrem(processing_queue, 1, message)

    async def stop(self):
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed successfully.")

        logger.info("Description Worker fully stopped.")
