import cv2
from loguru import logger
from guard.core.interfaces import CameraDriver

class RTSPCameraDriver(CameraDriver):
    def __init__(self, stream_url: str):
        self.stream_url = stream_url

    def stream(self):
        req_logger = logger.bind(stream=self.stream_url.rsplit("/", 1)[-1])

        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            req_logger.error("Error opening the camera stream")

            return

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    req_logger.warning("Error reading the camera frame")
                    break

                ret, buffer = cv2.imencode(".jpg", frame)

                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        finally:
            cap.release()
