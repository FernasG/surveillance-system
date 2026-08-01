import numpy as np
from ultralytics import YOLO

from guard.core.interfaces import ObjectDetector

class YOLODetector(ObjectDetector):
    def __init__(self):
        self.model = YOLO("yolo26n.pt", verbose=False)

    def detect(self, frame: np.ndarray) -> str:
        results = self.model.predict(frame, conf=0.30, verbose=False)[0]

        if results.boxes is None or len(results.boxes.cls) == 0:
            return ""

        unique_class_ids = set(results.boxes.cls.int().cpu().tolist())
        detected_classes = [self.model.names[cls_id] for cls_id in unique_class_ids]

        return ", ".join(detected_classes)
