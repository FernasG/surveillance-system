import cv2, threading
from fastapi import FastAPI
from contextlib import asynccontextmanager

from guard.core.entities import Query
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.drivers.usb_camera_driver import LogitechUSBDriver

from guard.pipeline.retrieval.retrieval_service import RetrievalService
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.acquisition.acquisition_service import AcquisitionService
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService

camera_driver = LogitechUSBDriver()
vectorizer = CLIPVectorizer()
sampler = MOG2FrameSampler()
store = ChromaDBStore()

# api nao pode instanciar pipeline de camera (se server cai, para a gravacao)
# 

inferer_service = InferenceService(vectorizer=vectorizer, store=store)
retrieval_service = RetrievalService(vectorizer=vectorizer, store=store)
preprocessor_service = PreprocessorService(sampler=sampler)
acquisition_service = AcquisitionService(camera_driver=camera_driver)

def run_video_pipeline():
    acquisition_service.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
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