from redis import asyncio as aioredis

from guard.core.entities import Settings, SEARCH_PRIORITY_CHANNEL
from guard.infrastructure.models.async_qwen_vlm import AsyncQwenVLM
from guard.infrastructure.models.yolo_detector import YOLODetector
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.infrastructure.messaging.redis_event_queue_worker import RedisEventQueueWorker
from guard.infrastructure.messaging.search_priority_listener import SearchPriorityListener
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.database.sqlite_analytics_store import SqliteAnalyticsStore
from guard.infrastructure.database.sqlite_store import SqliteDatabase

from guard.pipeline.description.description_service import DescriptionService
from guard.pipeline.acquisition.acquisition_service import AcquisitionService

class DescriptionWorkerContainer:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.redis_client = None
        self.priority_redis_client = None
        self.vlm = None
        self.priority_listener = None
        self.queue_worker = None

    async def initialize(self):
        self.vlm = AsyncQwenVLM()
        yolo_detector = YOLODetector()
        prompt_manager = PromptManager()

        self.redis_client = aioredis.Redis(host=self.settings.redis_host, port=self.settings.redis_port)
        self.priority_redis_client = aioredis.Redis(host=self.settings.redis_host, port=self.settings.redis_port)

        store = ChromaDBStore(host=self.settings.database_host, port=self.settings.database_port)

        database = SqliteDatabase()
        analytics_repo = SqliteAnalyticsStore(database)

        acquisition_service = AcquisitionService()

        self.priority_listener = SearchPriorityListener(self.priority_redis_client, SEARCH_PRIORITY_CHANNEL)

        description_service = DescriptionService(
            acquisition_service=acquisition_service,
            object_detector=yolo_detector,
            vlm=self.vlm,
            prompt_manager=prompt_manager,
            store=store,
            analytics_store=analytics_repo,
            search_active=self.priority_listener.search_active,
        )

        self.queue_worker = RedisEventQueueWorker(
            self.redis_client,
            description_service,
            self.priority_listener,
        )

    async def shutdown(self):
        if self.vlm:
            await self.vlm.aclose()

        if self.priority_redis_client:
            await self.priority_redis_client.close()
