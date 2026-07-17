import numpy as np
from ultralytics import YOLO

from guard.core.entities import DetectionObject
from guard.core.interfaces import ObjectDetector

class YOLODetector(ObjectDetector):
    def __init__(self):
        self.model = YOLO("yolo26n.pt", verbose=False)

    def detect(self, frame: np.ndarray) -> list[DetectionObject]:
        response: list[DetectionObject] = []

        results = self.model(frame)

        for result in results:
            names = result.names 
            
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = names[class_id]
                class_conf = box.conf[0]

                response.append(
                    DetectionObject(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=class_conf
                    )
                )

        return response
    
    def track(self, frame: np.ndarray) -> tuple[set[int], str]:
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        current_track_ids = set()
        detected_classes = set()

        if results.boxes is not None and results.boxes.id is not None:
            current_track_ids = set(results.boxes.id.int().cpu().tolist())
            
            for cls_id in results.boxes.cls:
                detected_classes.add(self.model.names[int(cls_id)])
        
        detected_str = ", ".join(list(detected_classes))

        return current_track_ids, detected_str
