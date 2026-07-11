import cv2
from loguru import logger
from guard.core.interfaces import CameraDriver

class LogitechUSBDriver(CameraDriver):
    def __init__(self, device_path="/dev/video0"):
        self.device_path = device_path

    def stream(self):
        req_logger = logger.bind(device_path=self.device_path)

        cap = cv2.VideoCapture(self.device_path)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            req_logger.error("Error opening the camera")

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