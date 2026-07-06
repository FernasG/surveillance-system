from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header, Request, Response
from fastapi.responses import StreamingResponse

from guard.core.entities import VideoListResponse, VideoItem
from guard.core.services.video_service import VideoService

router = APIRouter(prefix="/videos", tags=["Videos"])

def get_video_service() -> VideoService:
    return VideoService()

@router.get("/", response_model=VideoListResponse)
async def list_videos(request: Request, service: VideoService = Depends(get_video_service)):
    video_files = service.list_videos()
    base_url = str(request.base_url).rstrip("/")
    
    formatted_videos = [
        VideoItem(
            video_name=name,
            thumbnail_url=f"{base_url}/videos/{name}/thumbnail"
        )
        for name in video_files
    ]
    return {"videos": formatted_videos}

@router.get("/{video_name}")
async def get_video_stream(video_name: str, range: Optional[str] = Header(None), service: VideoService = Depends(get_video_service)):
    try:
        video_path = service.get_video_path(video_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_size = video_path.stat().st_size
    start = 0
    end = file_size - 1
    status_code = 200

    if range:
        status_code = 206
        range_value = range.replace("bytes=", "")

        try:
            start_str, end_str = range_value.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Header Range")

        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(status_code=416, detail=f"Requested range not met. File size: {file_size}")

    content_length = end - start + 1
    
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4",
    }

    return StreamingResponse(
        service.stream_video_chunk(video_path, start, end),
        status_code=status_code,
        headers=headers
    )

@router.get("/{video_name}/thumbnail")
async def get_video_thumbnail(video_name: str, service: VideoService = Depends(get_video_service)):
    try:
        image_bytes = service.extract_frame_as_jpeg(video_name)

        return Response(content=image_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))