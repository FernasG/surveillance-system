import asyncio
import cv2
import numpy as np
from PIL import Image
from typing import Optional
from loguru import logger

from guard.core.interfaces import ObjectDetector, VectorStoreInterface, AnalyticsStoreInterface, AsyncVLMInterface
from guard.core.entities import VLMMessage
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.pipeline.acquisition.acquisition_service import AcquisitionService

class DescriptionService:
    def __init__(
        self,
        acquisition_service: AcquisitionService,
        object_detector: ObjectDetector,
        vlm: AsyncVLMInterface,
        prompt_manager: PromptManager,
        store: VectorStoreInterface,
        analytics_store: AnalyticsStoreInterface,
        search_active: Optional[asyncio.Event] = None,
    ):
        self.acquisition_service = acquisition_service
        self.object_detector = object_detector
        self.vlm = vlm
        self.prompt_manager = prompt_manager
        self.store = store
        self.analytics_store = analytics_store
        self.search_active = search_active

        self.TARGET_SIZE = 640
        self.current_vlm_task: Optional[asyncio.Task] = None

    async def process(self, event_id: str, video_path: str, elapsed_ms: int) -> Optional[bool]:
        req_logger = logger.bind(event_id=event_id, video_path=video_path)

        frame = self._extract_frame(video_path, elapsed_ms)

        if frame is None:
            req_logger.warning("Failed to extract representative frame for event; dropping")
            return False

        objects = self.object_detector.detect(frame)
        messages, image = self._setup_vlm_params(frame)

        task = asyncio.create_task(self.vlm.generate(messages, [image]))
        self.current_vlm_task = task

        if self.search_active is not None and self.search_active.is_set():
            task.cancel()

        try:
            response = await task
        except asyncio.CancelledError:
            req_logger.info("Description generation cancelled in favor of a search request")
            return None
        finally:
            self.current_vlm_task = None

        description = response.content

        self.store.update_event(event_id, description=description, objects=objects)
        self.analytics_store.update_event_analytics(event_id, objects_detected=objects)

        return True

    def cancel_in_flight(self) -> None:
        if self.current_vlm_task and not self.current_vlm_task.done():
            self.current_vlm_task.cancel()

    def _extract_frame(self, video_path: str, elapsed_ms: int) -> Optional[np.ndarray]:
        cap = self.acquisition_service.get_video(video_path)

        try:
            if not cap.isOpened():
                return None

            cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)
            success, frame = cap.read()

            return frame if success else None
        finally:
            cap.release()

    def _setup_vlm_params(self, frame: np.ndarray) -> tuple[list[VLMMessage], Image.Image]:
        height, width, _ = frame.shape
        scale = self.TARGET_SIZE / max(width, height)
        new_w, new_h = int(width * scale), int(height * scale)

        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        prompt_text = self.prompt_manager.build(prompt_name="image_description")
        messages: list[VLMMessage] = [VLMMessage(role="user", content=prompt_text)]

        return messages, pil_img
