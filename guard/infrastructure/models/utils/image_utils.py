import io
import cv2
import base64
import numpy as np
from PIL.Image import Image

def pil_to_base64(pil_image: Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def cv2_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', frame, encode_param)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')