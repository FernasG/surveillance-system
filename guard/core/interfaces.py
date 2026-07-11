import cv2
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from guard.core.entities import VideoFrame, VectorEmbedding, VLMResponse, VLMMessage, User

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

class VLMInterface(ABC):
    @abstractmethod
    def generate(self, messages: list[VLMMessage], images: list = None, format_response: str = None) -> VLMResponse:
        pass

class VectorStoreInterface(ABC):
    @abstractmethod
    def save(self, embedding: VectorEmbedding) -> bool:
        pass

    @abstractmethod
    def save_batch(self, embeddings: list[VectorEmbedding]) -> bool:
        pass

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> dict:
        pass

class CameraDriver(ABC):
    @abstractmethod
    def stream(self):
        pass

class IAuthRepository(ABC):
    @abstractmethod
    def get_user(self, username: str) -> Optional[User]:
        pass

    @abstractmethod
    def save_user(self, user: User) -> None:
        pass