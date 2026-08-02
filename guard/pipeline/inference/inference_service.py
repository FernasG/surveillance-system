import uuid
import numpy as np
from typing import Optional
from loguru import logger
from datetime import datetime, timezone

from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, AnalyticsStoreInterface
from guard.core.entities import VideoFrame
from guard.infrastructure.messaging.redis_event_publisher import RedisEventPublisher

class InferenceService:
    def __init__(
        self,
        vectorizer: VectorizerInterface,
        analytics_store: AnalyticsStoreInterface,
        store: VectorStoreInterface,
        event_publisher: RedisEventPublisher,
    ):
        self.analytics_store = analytics_store
        self.vectorizer = vectorizer
        self.store = store
        self.event_publisher = event_publisher

        self.BATCH_SIZE = 8

        self.TIME_THRESHOLD_MS = 5000
        self.SIMILARITY_THRESHOLD = 0.82
        self.ALPHA_EMA = 0.30
        self.SPATIAL_OVERLAP_THRESHOLD = 0.10

        self.current_event_id = None
        self.event_centroid = None
        self.last_event_epoch_ms = None
        self.active_motion_bbox = None

        self.best_frame_video_path = None
        self.best_frame_elapsed_ms = None
        self.best_frame_motion_area = -1.0

    def inferer(self, frames: list[VideoFrame]):
        req_logger = logger.bind(frames_count=len(frames))

        try:
            req_logger.info("Running batch inference and event clustering on video frames")

            if not frames:
                return

            self._close_if_idle(frames[0])

            batches = [
                frames[i:i + self.BATCH_SIZE]
                for i in range(0, len(frames), self.BATCH_SIZE)
            ]

            for batch in batches:
                vectors = self.vectorizer.encode_batch_images(batch)
                batch_closures: list[dict] = []

                for idx, frame in enumerate(batch):
                    current_vector = vectors[idx].embeddings
                    current_epoch_ms = self._get_absolute_epoch_ms(frame)
                    current_motion_bbox = getattr(frame, "motion_bbox", None)
                    motion_area = self._bbox_area(current_motion_bbox)

                    is_new_event = True
                    similarity = 0.0

                    if self.current_event_id is not None:
                        time_diff = current_epoch_ms - self.last_event_epoch_ms

                        similarity = float(np.dot(current_vector, self.event_centroid))

                        if time_diff <= self.TIME_THRESHOLD_MS:
                            has_semantic_continuity = similarity >= self.SIMILARITY_THRESHOLD

                            has_spatial_continuity = self._iou(
                                current_motion_bbox, self.active_motion_bbox
                            ) >= self.SPATIAL_OVERLAP_THRESHOLD

                            if has_semantic_continuity or has_spatial_continuity:
                                is_new_event = False

                    if is_new_event:
                        closure = self._close_current_event()

                        if closure:
                            batch_closures.append(closure)

                        self.current_event_id = str(uuid.uuid4())
                        req_logger.debug(f"New event detected: {self.current_event_id} at {current_epoch_ms}ms")

                        self.event_centroid = current_vector.copy()
                        self.active_motion_bbox = current_motion_bbox

                        self.best_frame_video_path = frame.video_path
                        self.best_frame_elapsed_ms = frame.elapsed_ms
                        self.best_frame_motion_area = motion_area

                        self.analytics_store.save_event_analytics(
                            event_id=self.current_event_id,
                            objects_detected="",
                            timestamp=self._get_event_timestamp(frame)
                        )
                    else:
                        self.event_centroid = (self.ALPHA_EMA * current_vector) + ((1.0 - self.ALPHA_EMA) * self.event_centroid)
                        self.event_centroid = self.event_centroid / np.linalg.norm(self.event_centroid)
                        self.active_motion_bbox = current_motion_bbox

                        if motion_area > self.best_frame_motion_area:
                            self.best_frame_motion_area = motion_area
                            self.best_frame_video_path = frame.video_path
                            self.best_frame_elapsed_ms = frame.elapsed_ms

                    if vectors[idx].metadata is None:
                        vectors[idx].metadata = {}

                    vectors[idx].metadata["event_id"] = self.current_event_id
                    vectors[idx].metadata["description"] = "pending"
                    vectors[idx].metadata["is_keyframe"] = is_new_event
                    vectors[idx].metadata["objects"] = ""

                    self.last_event_epoch_ms = current_epoch_ms

                self.store.save_batch(vectors)

                for closure in batch_closures:
                    self.event_publisher.publish_event_closed(**closure)
        except Exception:
            req_logger.exception("Failed to execute inference or save vector batches")
            raise

    def close_open_event(self) -> None:
        closure = self._close_current_event()

        if closure:
            logger.bind(event_id=closure["event_id"]).info("Flushing open event on shutdown")
            self.event_publisher.publish_event_closed(**closure)

    def _close_if_idle(self, next_frame: VideoFrame) -> None:
        if self.current_event_id is None:
            return

        first_epoch_ms = self._get_absolute_epoch_ms(next_frame)

        if first_epoch_ms - self.last_event_epoch_ms > self.TIME_THRESHOLD_MS:
            closure = self._close_current_event()

            if closure:
                self.event_publisher.publish_event_closed(**closure)

    def _close_current_event(self) -> Optional[dict]:
        if self.current_event_id is None:
            return None

        closure = {
            "event_id": self.current_event_id,
            "video_path": self.best_frame_video_path,
            "elapsed_ms": self.best_frame_elapsed_ms,
        }

        self.current_event_id = None
        self.event_centroid = None
        self.active_motion_bbox = None
        self.best_frame_video_path = None
        self.best_frame_elapsed_ms = None
        self.best_frame_motion_area = -1.0

        return closure

    def _bbox_area(self, bbox: Optional[tuple]) -> float:
        if bbox is None:
            return 0.0

        x1, y1, x2, y2 = bbox

        return max(0, x2 - x1) * max(0, y2 - y1)

    def _iou(self, box_a: tuple, box_b: tuple) -> float:
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _get_absolute_epoch_ms(self, frame: VideoFrame) -> float:
        return frame.timestamp * 1000 + frame.elapsed_ms

    def _get_event_timestamp(self, frame: VideoFrame) -> str:
        event_epoch_seconds = frame.timestamp + (frame.elapsed_ms / 1000.0)

        return datetime.fromtimestamp(event_epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
