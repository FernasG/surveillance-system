import cv2
import uuid
import numpy as np
from PIL import Image
from loguru import logger
from datetime import datetime, timezone

from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, VLMInterface, ObjectDetector
from guard.core.entities import VideoFrame, VLMMessage
from guard.core.interfaces import AnalyticsStoreInterface

class InferenceService:
    def __init__(
        self,
        vectorizer: VectorizerInterface,
        prompt_manager: PromptManager,
        object_detector: ObjectDetector,
        analytics_store: AnalyticsStoreInterface,
        store: VectorStoreInterface,
        vlm: VLMInterface,
    ):
        self.object_detector = object_detector
        self.analytics_store = analytics_store
        self.prompt_manager = prompt_manager
        self.vectorizer = vectorizer
        self.store = store
        self.vlm = vlm
        
        self.BATCH_SIZE = 8
        self.TARGET_SIZE = 640

        self.TIME_THRESHOLD_MS = 5000 
        self.SIMILARITY_THRESHOLD = 0.82
        self.PEAK_ESSENCE_THRESHOLD = 0.94
        self.ALPHA_EMA = 0.30

        self.current_event_id = None
        self.event_centroid = None
        self.event_frame_count = 0
        self.last_timestamp_ms = 0.0
        self.current_event_description = "Description unavailable"

    def inferer(self, frames: list[VideoFrame]):
        req_logger = logger.bind(frames_count=len(frames))

        try:
            req_logger.info("Running batch inference and event clustering on video frames")

            batches = [
                frames[i:i + self.BATCH_SIZE]
                for i in range(0, len(frames), self.BATCH_SIZE)
            ]

            for batch in batches:
                vectors = self.vectorizer.encode_batch_images(batch)

                for idx, frame in enumerate(batch):
                    current_vector = vectors[idx].embeddings
                    current_time_ms = getattr(frame, "elapsed_ms")

                    is_new_event = True
                    similarity = 0.0

                    current_track_ids, detected_str = self.object_detector.track(frame.data)

                    if self.current_event_id is not None:
                        time_diff = current_time_ms - self.last_timestamp_ms
                        
                        similarity = float(np.dot(current_vector, self.event_centroid))

                        if time_diff <= self.TIME_THRESHOLD_MS:
                            has_common_tracks = bool(self.active_track_ids.intersection(current_track_ids))
                            
                            has_semantic_continuity = similarity >= self.SIMILARITY_THRESHOLD

                            if has_common_tracks:
                                is_new_event = False
                                self.active_track_ids.update(current_track_ids)
                            elif len(current_track_ids) == 0 and len(self.active_track_ids) > 0:
                                is_new_event = False
                            elif has_semantic_continuity:
                                is_new_event = False
                                self.active_track_ids.update(current_track_ids)

                    if is_new_event:
                        self.current_event_id = str(uuid.uuid4())
                        req_logger.debug(f"New event detected: {self.current_event_id} at {current_time_ms}ms")
                        
                        self.event_centroid = current_vector.copy()
                        self.highest_essence_score = 0.0
                        self.active_track_ids = current_track_ids

                        self.current_event_description = self._get_image_description(frame)

                        self.analytics_store.save_event_analytics(
                            event_id=self.current_event_id,
                            objects_detected=detected_str,
                            timestamp=self._get_event_timestamp(frame)
                        )
                    else:
                        self.event_centroid = (self.ALPHA_EMA * current_vector) + ((1.0 - self.ALPHA_EMA) * self.event_centroid)
                        self.event_centroid = self.event_centroid / np.linalg.norm(self.event_centroid)

                        if similarity > self.PEAK_ESSENCE_THRESHOLD and similarity > self.highest_essence_score:
                            self.highest_essence_score = similarity
                            req_logger.info(f"Peak action found (sim: {similarity:.3f}). Refining VLM description.")
                            
                            self.current_event_description = self._get_image_description(frame)

                    if vectors[idx].metadata is None:
                        vectors[idx].metadata = {}
                    
                    vectors[idx].metadata["event_id"] = self.current_event_id
                    vectors[idx].metadata["description"] = self.current_event_description
                    vectors[idx].metadata["is_keyframe"] = is_new_event
                    vectors[idx].metadata["objects"] = detected_str

                    self.last_timestamp_ms = current_time_ms

                self.store.save_batch(vectors)
        except Exception:
            req_logger.exception("Failed to execute inference or save vector batches")
            raise

    def _get_event_timestamp(self, frame: VideoFrame) -> str:
        event_epoch_seconds = frame.timestamp + (frame.elapsed_ms / 1000.0)

        return datetime.fromtimestamp(event_epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _get_image_description(self, frame: VideoFrame) -> str:
        try:
            messages, image = self._setup_vlm_params(frame.data)
            response = self.vlm.generate(messages, [image])

            return response.content
        except Exception as e:
            logger.warning(f"Failed to generate description for frame: {e}")
            return "Description unavailable"

    def _setup_vlm_params(self, frame: np.ndarray) -> tuple[list[VLMMessage], list[Image.Image]]:
        height, width, _ = frame.shape
        scale = self.TARGET_SIZE / max(width, height)
        new_w, new_h = int(width * scale), int(height * scale)

        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        prompt_text = self.prompt_manager.build(prompt_name="image_description")
        messages: list[VLMMessage] = [VLMMessage(role="user", content=prompt_text)]

        return messages, pil_img
