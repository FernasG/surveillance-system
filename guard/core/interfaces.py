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

class VectorStoreInterface(ABC):
    @abstractmethod
    def save(self, embedding: VectorEmbedding) -> bool:
        pass