import asyncio, contextlib, json, logging
from redis import asyncio as aioredis
from guard.core.entities import QueueMessage
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.acquisition.acquisition_service import AcquisitionService
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

logger = logging.getLogger(__name__)

class RedisQueueWorker:
    def __init__(self, redis_client: aioredis.Redis, acquisition_service: AcquisitionService, inference_service: InferenceService, preprocessor_service: PreprocessorService):
        self.inference_service = inference_service
        self.acquisition_service = acquisition_service
        self.preprocessor_service = preprocessor_service
        self._redis_client = redis_client

    async def start(self, queue_name: str):
        logger.info("Starting Redis Worker (Async)")

        processing_queue = f"{queue_name}:processing"

        try:
            await self._reclaim_pending(queue_name, processing_queue)

            while True:
                message = await self._redis_client.blmove(queue_name, processing_queue, 0, "RIGHT", "LEFT")

                if not message:
                    continue

                work = asyncio.create_task(self._process_message(message))

                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    if not work.done():
                        with contextlib.suppress(Exception):
                            await work

                    await self._ack(processing_queue, message)
                    raise
                except json.JSONDecodeError:
                    logger.error("Failed to decode Redis message JSON.")
                except Exception as e:
                    logger.error(f"Unexpected error in worker pipeline: {str(e)}", exc_info=True)

                await self._ack(processing_queue, message)

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled, shutting down")

            self.inference_service.close_open_event()
        finally:
            await self.stop()

    async def _process_message(self, message) -> None:
        data = json.loads(message)
        queue_message = QueueMessage(**data)

        logger.info(f"Processing new video: {queue_message.video_path}")

        video = self.acquisition_service.get_video(queue_message.video_path)

        frames = await asyncio.to_thread(self.preprocessor_service.process, queue_message, video)
        await asyncio.to_thread(self.inference_service.inferer, frames)

    async def _reclaim_pending(self, queue_name: str, processing_queue: str) -> None:
        reclaimed = 0

        while await self._redis_client.lmove(processing_queue, queue_name, "RIGHT", "RIGHT"):
            reclaimed += 1

        if reclaimed:
            logger.info(f"Reclaimed {reclaimed} in-flight video segment(s) from a previous run")

    async def _ack(self, processing_queue: str, message) -> None:
        await self._redis_client.lrem(processing_queue, 1, message)

    async def stop(self):
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed successfully.")

        logger.info("Redis Worker fully stopped.")
