from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request, Response, Query, Header
from fastapi.responses import StreamingResponse
from datetime import date

from guard.api.routers.auth_router import get_current_user_token_data
from guard.core.interfaces import CameraDriver
from guard.core.entities import VideoItem, VideoListResponse
from guard.core.services.video_service import VideoService
from guard.infrastructure.drivers.usb_camera_driver import LogitechUSBDriver

router = APIRouter(prefix="/videos", tags=["Videos"])

def get_video_service() -> VideoService:
    return VideoService()

def get_camera_driver() -> CameraDriver:
    return LogitechUSBDriver(device_path="/dev/video0")

@router.get("/", response_model=VideoListResponse, dependencies=[Depends(get_current_user_token_data)])
async def list_videos(
    request: Request, 
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    service: VideoService = Depends(get_video_service)
):
    video_files, total_items = service.list_videos(
        page=page, 
        size=size, 
        start_date=start_date, 
        end_date=end_date
    )
    
    base_url = str(request.base_url).rstrip("/")
    
    formatted_videos = [
        VideoItem(
            video_name=name,
            thumbnail_url=f"{base_url}/videos/{name}/thumbnail"
        )
        for name in video_files
    ]
    
    total_pages = (total_items + size - 1) // size

    return {
        "videos": formatted_videos,
        "total_items": total_items,
        "page": page,
        "size": size,
        "total_pages": total_pages
    }

@router.get("/live", dependencies=[Depends(get_current_user_token_data)])
async def stream_live_camera(camera_driver: CameraDriver = Depends(get_camera_driver)):
    return StreamingResponse(camera_driver.stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/{video_name}/playback", dependencies=[Depends(get_current_user_token_data)])
async def stream_recorded_video(video_name: str, range: Optional[str] = Header(None), service: VideoService = Depends(get_video_service)):
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

@router.get("/{video_name}/thumbnail", dependencies=[Depends(get_current_user_token_data)])
async def get_video_thumbnail(video_name: str, service: VideoService = Depends(get_video_service)):
    try:
        image_bytes = service.extract_frame_as_jpeg(video_name)

        return Response(content=image_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
