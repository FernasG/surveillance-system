import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, Union
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
class VLMMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: list[dict]

class Query(BaseModel):
    text: str = Field(min_length=3)

class Settings(BaseSettings):
    env: str
    clip_model: str
    database_host: str
    database_port: str
    videos_dir: str
    redis_host: str
    redis_port: str
    redis_queue_name: str

    class Config:
        env_file = ".env"
