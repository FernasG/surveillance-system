import numpy as np
from guard.core.interfaces import VectorizerInterface
from guard.core.entities import VideoFrame, VectorEmbedding

class CLIPVectorizer(VectorizerInterface):
    def __init__(self):
        super().__init__()

    def encode_image(self, frame: VideoFrame) -> VectorEmbedding:
        pass
    
    def encode_text(self, query_text: str) -> np.ndarray:
        pass
    
    def encode_batch_images(self, frames: list[VideoFrame]) -> list[VectorEmbedding]:
        if not frames:
            return []

        return [
            VectorEmbedding(
                embeddings=vector,
                metadata={
                    "elapsed_ms": frame.elapsed_ms,
                    "frame_index": frame.frame_index,
                    "video_path": frame.video_path,
                    "timestamp": frame.timestamp
                }) 
            for vector, frame in zip([], frames)
        ]