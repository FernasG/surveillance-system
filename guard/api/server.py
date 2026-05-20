from fastapi import FastAPI, Depends, Request

from guard.core.entities import Query
from guard.api.lifespan import app_lifespan
from guard.pipeline.retrieval.retrieval_service import RetrievalService

app = FastAPI(lifespan=app_lifespan)

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.state.retrieval_service

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "OK"}

@app.post("/query")
async def query(query: Query, retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    return retrieval_service.search_by_text(query.text)