import numpy as np
from pydantic import BaseModel, Field
from dataclasses import dataclass
from pydantic_settings import BaseSettings

@dataclass
class VideoFrame:
    timestamp: float
    data: np.ndarray
    source_id: str

@dataclass
class VectorEmbedding:
    embeddings: np.ndarray
    metadata: dict

class Query(BaseModel):
    text: str = Field(min_length=3)

class Settings(BaseSettings):
    clip_model: str
    database_host: str
    database_port: str
    videos_dir: str
    redis_host: str
    redis_port: str
    redis_queue_name: str

    class Config:
        env_file = ".env"