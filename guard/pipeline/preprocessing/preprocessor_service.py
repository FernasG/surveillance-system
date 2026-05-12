import cv2
from guard.core.interfaces import VideoFrameSampler

class PreprocessorService():
    def __init__(self, sampler: VideoFrameSampler):
        self.sampler = sampler

    def process(self, video: cv2.VideoCapture):
        return self.sampler.get_frames(video)