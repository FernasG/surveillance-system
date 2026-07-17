import asyncio
from redis import asyncio as aioredis

from guard.core.entities import Settings
from guard.infrastructure.models.qwen_vlm import QwenVLM
from guard.infrastructure.models.gemma_vlm import GemmaVLM
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.messaging.queue_worker import RedisQueueWorker
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.infrastructure.models.yolo_detector import YOLODetector

from guard.core.services.retrieval_service import RetrievalService
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.acquisition.acquisition_service import AcquisitionService
from guard.infrastructure.database.redis_auth_store import RedisAuthStore
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService
from guard.core.services.auth_service import AuthService

class ApplicationContainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        
        self.vectorizer = None
        self.redis_client = None
        self.auth_service = None
        self.retrieval_service = None
        self._worker_task = None

    async def initialize(self):
        self.vectorizer = CLIPVectorizer()
        prompt_manager = PromptManager()
        yolo_detector = YOLODetector()
        sampler = MOG2FrameSampler()
        gemma_vlm = GemmaVLM()
        qwen_vlm = QwenVLM()

        self.redis_client = aioredis.Redis(host=self.settings.redis_host, port=self.settings.redis_port)
        store = ChromaDBStore(host=self.settings.database_host, port=self.settings.database_port)

        auth_repo = RedisAuthStore(self.redis_client)
        self.auth_service = AuthService(auth_repo)
        await self.auth_service.initialize_admin()

        acquisition_service = AcquisitionService()
        preprocessor_service = PreprocessorService(sampler=sampler)
        inference_service = InferenceService(
            vectorizer=self.vectorizer, object_detector=yolo_detector,
            store=store, vlm=qwen_vlm, prompt_manager=prompt_manager
        )
        self.retrieval_service = RetrievalService(vectorizer=self.vectorizer, store=store, vlm=gemma_vlm, prompt_manager=prompt_manager)

        queue_worker = RedisQueueWorker(
            self.redis_client, 
            acquisition_service, 
            inference_service, 
            preprocessor_service
        )
        
        self._worker_task = asyncio.create_task(queue_worker.start(self.settings.redis_queue_name))

    async def shutdown(self):
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.vectorizer:
            del self.vectorizer