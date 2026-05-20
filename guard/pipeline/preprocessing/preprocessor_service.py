import cv2
from guard.core.entities import QueueMessage, VideoFrame
from guard.core.interfaces import VideoFrameSampler

class PreprocessorService():
    def __init__(self, sampler: VideoFrameSampler):
        self.sampler = sampler

    def process(self, message: QueueMessage, video: cv2.VideoCapture) -> list[VideoFrame]:
        return self.sampler.get_frames(message, video)