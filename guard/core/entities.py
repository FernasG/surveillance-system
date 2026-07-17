import numpy as np
from typing import Literal, List
from pydantic import BaseModel, Field
from dataclasses import dataclass
from pydantic_settings import BaseSettings

@dataclass
class VideoFrame:
    timestamp: float
    video_path: str
    elapsed_ms: int
    frame_index: int
    data: np.ndarray

@dataclass
class VectorEmbedding:
    embeddings: np.ndarray
    metadata: dict

@dataclass
class QueueMessage:
    timestamp: int
    video_path: str
    created_at: str

@dataclass
class VLMResponse:
    role: str
    content: str

@dataclass
class VLMMessage:
    role: Literal["user", "assistant", "system"]
    content: str

@dataclass
class DetectionObject:
    class_id: int
    class_name: str
    confidence: float

class Query(BaseModel):
    text: str = Field(min_length=3)

class User(BaseModel):
    username: str
    hashed_password: str
    is_admin: bool = False

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class VideoItem(BaseModel):
    video_name: str
    thumbnail_url: str

class VideoListResponse(BaseModel):
    videos: List[VideoItem]
    page: int
    size: int
    total_items: int
    total_pages: int

class Settings(BaseSettings):
    env: str
    clip_model: str
    database_host: str
    database_port: str
    videos_dir: str
    redis_host: str
    redis_port: str
    redis_queue_name: str
    generation_model: str
    generation_server_url: str
    image_model: str
    image_server_url: str
    access_token_expire_minutes: float
    secret_key: str
    algorithm: str

    class Config:
        env_file = ".env"
