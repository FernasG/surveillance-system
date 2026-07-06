from fastapi import APIRouter, Depends, Request
from guard.core.entities import Query
from guard.api.routers.auth_router import get_current_user_token_data
from guard.core.services.retrieval_service import RetrievalService

router = APIRouter(tags=["Search / Retrieval"])

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.state.retrieval_service

@router.post("/query", dependencies=[Depends(get_current_user_token_data)])
async def query(query: Query, retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    return retrieval_service.search_by_text(query.text)
