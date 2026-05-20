import asyncio, json, logging
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
        
        try:
            while True:
                try:
                    result = await self._redis_client.brpop(queue_name)

                    if not result:
                        continue
                        
                    _, message = result
                    data = json.loads(message)

                    queue_message = QueueMessage(**data)
                    
                    logger.info(f"Processing new video: {queue_message.video_path}")
                    
                    video = self.acquisition_service.get_video(queue_message.video_path)
                    frames = self.preprocessor_service.process(queue_message, video)
                    self.inference_service.inferer(frames)
                    
                except json.JSONDecodeError:
                    logger.error("Failed to decode Redis message JSON.")
                except Exception as e:
                    logger.error(f"Unexpected error in worker pipeline: {str(e)}", exc_info=True)

        except asyncio.CancelledError:
            logger.error(f"Unexpected error in worker pipeline: {str(e)}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed successfully.")

        logger.info("Redis Worker fully stopped.")