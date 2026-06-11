from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from guard.core.entities import Query
from guard.api.lifespan import app_lifespan
from guard.api.middlewares import RequestIdMiddleware
from guard.infrastructure.logging.logger_config import setup_logging
from guard.pipeline.retrieval.retrieval_service import RetrievalService
from guard.core.entities import Settings

settings = Settings()

IS_PRODUCTION = settings.env == "production"

setup_logging(json_format=IS_PRODUCTION)

app = FastAPI(title="Pi Guard", lifespan=app_lifespan)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.state.retrieval_service

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "OK"}

@app.post("/query")
async def query(query: Query, retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    return retrieval_service.search_by_text(query.text)