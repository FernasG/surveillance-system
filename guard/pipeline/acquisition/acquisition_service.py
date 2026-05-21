import cv2
from loguru import logger

class AcquisitionService:
    def get_video(self, video_path: str) -> cv2.VideoCapture:
        req_logger = logger.bind(video_path=video_path)

        try:
            req_logger.info("Fetching video file for processing")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                req_logger.error("Failed to open video file or stream")

            return cap
        except Exception:
            req_logger.exception("Failed to fetch video")

            raise