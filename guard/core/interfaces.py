import cv2
import numpy as np
from abc import ABC, abstractmethod
from guard.core.entities import VideoFrame, VectorEmbedding


class VideoFrameSampler(ABC):
    @abstractmethod
    def get_frames(self, video: cv2.VideoCapture) -> list[VideoFrame]:
        pass

class VectorizerInterface(ABC):
    @abstractmethod
    def encode_image(self, frame: VideoFrame) -> VectorEmbedding:
        pass

    @abstractmethod
    def encode_text(self, query_text: str) -> np.ndarray:
        pass

    @abstractmethod
    def encode_batch_images(self, frames: list[VideoFrame]) -> list[VectorEmbedding]:
        pass

class VectorStoreInterface(ABC):
    @abstractmethod
    def save(self, embedding: VectorEmbedding, doc_id: str) -> bool:
        pass

    @abstractmethod
    def save_batch(self, embeddings: list[VectorEmbedding], doc_id: str) -> bool:
        pass

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        pass

class CameraDriver(ABC):
    @abstractmethod
    def start_recording(self, segment_time: int, folder_path: str):
        pass

    @abstractmethod
    def stop_recording(self):
        pass