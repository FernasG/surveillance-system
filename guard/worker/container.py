from redis import asyncio as aioredis

from guard.core.entities import Settings
from guard.infrastructure.models.qwen_vlm import QwenVLM
from guard.infrastructure.models.yolo_detector import YOLODetector
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.messaging.queue_worker import RedisQueueWorker
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.database.sqlite_analytics_store import SqliteAnalyticsStore
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.infrastructure.database.sqlite_store import SqliteDatabase

from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler
from guard.pipeline.acquisition.acquisition_service import AcquisitionService
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

class WorkerContainer:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.redis_client = None
        self.queue_worker = None

    async def initialize(self):
        vectorizer = CLIPVectorizer()
        prompt_manager = PromptManager()
        yolo_detector = YOLODetector()
        sampler = MOG2FrameSampler()
        qwen_vlm = QwenVLM()

        self.redis_client = aioredis.Redis(host=self.settings.redis_host, port=self.settings.redis_port)
        store = ChromaDBStore(host=self.settings.database_host, port=self.settings.database_port)

        database = SqliteDatabase()
        analytics_repo = SqliteAnalyticsStore(database)

        acquisition_service = AcquisitionService()
        preprocessor_service = PreprocessorService(sampler=sampler)
        inference_service = InferenceService(
            vectorizer=vectorizer, object_detector=yolo_detector,
            store=store, vlm=qwen_vlm, prompt_manager=prompt_manager,
            analytics_store=analytics_repo
        )

        self.queue_worker = RedisQueueWorker(
            self.redis_client,
            acquisition_service,
            inference_service,
            preprocessor_service
        )

    async def shutdown(self):
        # RedisQueueWorker.start() already closes the redis connection
        # in its own finally block once its task is cancelled.
        pass
