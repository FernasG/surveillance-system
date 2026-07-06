import os
import cv2
from pathlib import Path
from typing import List
from typing import Generator
from functools import lru_cache
from guard.core.entities import Settings

class VideoService:
    def __init__(self):
        settings = Settings()

        self.videos_dir = Path(settings.videos_dir)

    def list_videos(self) -> List[str]:
        if not self.videos_dir.exists() or not self.videos_dir.is_dir():
            return []
            
        valid_extensions = {".mp4", ".avi", ".mkv", ".mov"}
        return [
            file.name for file in self.videos_dir.iterdir() 
            if file.is_file() and file.suffix.lower() in valid_extensions
        ]

    def get_video_path(self, video_name: str) -> Path:
        video_path = self.videos_dir / video_name
        
        if not os.path.abspath(video_path).startswith(os.path.abspath(self.videos_dir)):
            raise ValueError("Invalid file path")
        
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video '{video_name}' not found")
        
        return video_path
    
    def stream_video_chunk(self, video_path: Path, start: int, end: int, chunk_size: int = 1024 * 1024) -> Generator[bytes, None, None]:
        with open(video_path, "rb") as video:
            video.seek(start)
            remaining = end - start + 1
            
            while remaining > 0:
                bytes_to_read = min(chunk_size, remaining)
                data = video.read(bytes_to_read)
            
                if not data:
                    break
            
                remaining -= len(data)
            
                yield data

    @lru_cache(maxsize=128)
    def extract_frame_as_jpeg(self, video_name: str) -> bytes:
        video_path = self.videos_dir / video_name
        
        if not os.path.abspath(video_path).startswith(os.path.abspath(self.videos_dir)):
            raise ValueError("Invalid path")
        if not video_path.exists():
            raise FileNotFoundError("Video not found")

        cap = cv2.VideoCapture(str(video_path))
        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError("The video frame could not be read")

        height, width, _ = frame.shape
        new_width = 320
        new_height = int((new_width / width) * height)
        frame = cv2.resize(frame, (new_width, new_height))

        success, encoded_image = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Error encoding image")

        return encoded_image.tobytes()