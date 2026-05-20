import cv2

class AcquisitionService:
    def get_video(self, video_path: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(video_path)
