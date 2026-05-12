import numpy as np
from dataclasses import dataclass, field

@dataclass
class VideoFrame:
    timestamp: float
    data: np.ndarray
    source_id: str

@dataclass
class VectorEmbedding:
    embeddings: np.ndarray
    metadata: dict
