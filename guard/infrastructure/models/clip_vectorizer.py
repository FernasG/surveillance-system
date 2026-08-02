import requests
import numpy as np
from guard.core.interfaces import VectorizerInterface
from guard.core.entities import VideoFrame, VectorEmbedding, Settings
from guard.infrastructure.models.utils.image_utils import cv2_to_base64

class CLIPVectorizer(VectorizerInterface):
    def __init__(self):
        super().__init__()

        settings = Settings()

        self.api_url = settings.clip_server_url
        self.request_timeout = 60

    def encode_image(self, frame: VideoFrame) -> VectorEmbedding:
        embeddings = self._encode_images([frame.data])[0]

        return VectorEmbedding(embeddings=embeddings, metadata={})

    def encode_text(self, query_text: str) -> np.ndarray:
        response = requests.post(
            f"{self.api_url}/encode/text",
            json={"text": query_text},
            timeout=self.request_timeout
        )
        response.raise_for_status()

        return np.array(response.json()["embedding"], dtype=np.float32)

    def encode_batch_images(self, frames: list[VideoFrame]) -> list[VectorEmbedding]:
        if not frames:
            return []

        embeddings = self._encode_images([frame.data for frame in frames])

        return [
            VectorEmbedding(
                embeddings=vector,
                metadata={
                    "elapsed_ms": frame.elapsed_ms,
                    "frame_index": frame.frame_index,
                    "video_path": frame.video_path,
                    "timestamp": frame.timestamp
                })
            for vector, frame in zip(embeddings, frames)
        ]

    def _encode_images(self, images: list[np.ndarray]) -> list[np.ndarray]:
        payload = {"images": [cv2_to_base64(image, quality=90) for image in images]}

        response = requests.post(
            f"{self.api_url}/encode/images",
            json=payload,
            timeout=self.request_timeout
        )
        response.raise_for_status()

        return [np.array(vector, dtype=np.float32) for vector in response.json()["embeddings"]]
