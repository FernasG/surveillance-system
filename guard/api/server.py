import asyncio
from redis import asyncio as aioredis
from fastapi import FastAPI, Depends, Request
from contextlib import asynccontextmanager

from guard.core.entities import Query, Settings
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.messaging.queue_worker import RedisQueueWorker

from guard.pipeline.retrieval.retrieval_service import RetrievalService
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.acquisition.acquisition_service import AcquisitionService
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorizer = CLIPVectorizer()
    sampler = MOG2FrameSampler()

    redis_client = aioredis.Redis(host=settings.redis_host, port=settings.redis_port)
    store = ChromaDBStore(host=settings.database_host, port=settings.database_port)

    acquisition_service = AcquisitionService()
    inference_service = InferenceService(vectorizer=vectorizer, store=store)
    retrieval_service = RetrievalService(vectorizer=vectorizer, store=store)
    preprocessor_service = PreprocessorService(sampler=sampler)

    queue_worker = RedisQueueWorker(redis_client, acquisition_service, inference_service, preprocessor_service)
    worker_task = asyncio.create_task(queue_worker.start(settings.redis_queue_name))
    
    yield {
        "vectorizer": vectorizer,
        "retrieval_service": retrieval_service
    }

    worker_task.cancel()

    await asyncio.gather(worker_task, return_exceptions=True)
    
    del vectorizer

app = FastAPI(lifespan=lifespan)

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.state.retrieval_service

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "OK"}

@app.post("/query")
async def query(query: Query, retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    return retrieval_service.search_by_text(query.text)