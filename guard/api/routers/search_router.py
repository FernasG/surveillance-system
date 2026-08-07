from typing import Optional
from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, HTTPException
from guard.core.entities import Query
from guard.api.routers.auth_router import get_current_user_token_data
from guard.core.services.retrieval_service import RetrievalService
from guard.infrastructure.models.utils.image_utils import bytes_to_cv2

router = APIRouter(tags=["Search / Retrieval"])

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.state.retrieval_service

@router.post("/query", dependencies=[Depends(get_current_user_token_data)])
def query(query: Query, retrieval_service: RetrievalService = Depends(get_retrieval_service)):
    return retrieval_service.search_by_text(query.text)

@router.post("/query/image", dependencies=[Depends(get_current_user_token_data)])
async def query_by_image(
    top_k: int = Form(5),
    image: Optional[UploadFile] = File(None),
    video_filename: Optional[str] = Form(None),
    timestamp_s: Optional[float] = Form(None),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    if image is not None:
        contents = await image.read()

        try:
            frame = bytes_to_cv2(contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return retrieval_service.search_by_image(frame, top_k=top_k)

    if video_filename is not None and timestamp_s is not None:
        try:
            return retrieval_service.search_by_video_frame(video_filename, timestamp_s, top_k=top_k)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=400, detail="Provide either an image file, or video_filename and timestamp_s")
