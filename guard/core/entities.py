import numpy as np
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

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