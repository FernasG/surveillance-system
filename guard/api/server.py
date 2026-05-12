import cv2, threading
from fastapi import FastAPI
from contextlib import asynccontextmanager

from guard.core.entities import Query
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.pipeline.retrieval.retrieval_service import RetrievalService
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

vectorizer = CLIPVectorizer()
sampler = MOG2FrameSampler()
store = ChromaDBStore()

inferer_service = InferenceService(vectorizer=vectorizer, store=store)
retrieval_service = RetrievalService(vectorizer=vectorizer, store=store)
preprocessor_service = PreprocessorService(sampler=sampler)

def run_video_pipeline():
    print("Iniciando Pipeline de Vídeo...")
    video = cv2.VideoCapture("/app/videos/video.mp4")
    
    frames = preprocessor_service.process(video)
    inferer_service.inferer(frames)
    print("Pipeline de Vídeo Finalizado.")

@asynccontextmanager
async def lifespan():
    thread = threading.Thread(target=run_video_pipeline)
    thread.start()
    
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "OK"}

@app.post("/query")
async def query(query: Query):
    return retrieval_service.search_by_text(query.text)