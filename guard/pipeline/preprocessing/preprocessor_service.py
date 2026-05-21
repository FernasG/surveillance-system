import cv2
from loguru import logger
from guard.core.entities import QueueMessage, VideoFrame
from guard.core.interfaces import VideoFrameSampler

class PreprocessorService:
    def __init__(self, sampler: VideoFrameSampler):
        self.sampler = sampler

    def process(self, message: QueueMessage, video: cv2.VideoCapture) -> list[VideoFrame]:
        log_context = {
            "timestamp": message.timestamp,
            "video_path": message.video_path,
            "created_at": str(message.created_at)
        }
        req_logger = logger.bind(queue_message=log_context)

        try:
            req_logger.info("Sampling video frames")

            return self.sampler.get_frames(message, video)
        except Exception:
            req_logger.exception("Failed sample and process video frames")

            raise