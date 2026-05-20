import asyncio
from redis import asyncio as aioredis

from guard.core.entities import Settings
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.messaging.queue_worker import RedisQueueWorker

from guard.pipeline.retrieval.retrieval_service import RetrievalService
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.acquisition.acquisition_service import AcquisitionService
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

class ApplicationContainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        
        self.vectorizer = None
        self.redis_client = None
        self.retrieval_service = None
        self._worker_task = None

    async def initialize(self):
        self.vectorizer = CLIPVectorizer()
        sampler = MOG2FrameSampler()
        
        self.redis_client = aioredis.Redis(host=self.settings.redis_host, port=self.settings.redis_port)
        store = ChromaDBStore(host=self.settings.database_host, port=self.settings.database_port)

        acquisition_service = AcquisitionService()
        inference_service = InferenceService(vectorizer=self.vectorizer, store=store)
        self.retrieval_service = RetrievalService(vectorizer=self.vectorizer, store=store)
        preprocessor_service = PreprocessorService(sampler=sampler)

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