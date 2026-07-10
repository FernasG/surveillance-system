import cv2
import numpy as np
from PIL import Image
from loguru import logger

from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, VLMInterface
from guard.core.entities import VideoFrame, VLMMessage

class InferenceService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface, vlm: VLMInterface, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.vectorizer = vectorizer
        self.store = store
        self.vlm = vlm
        self.BATCH_SIZE = 8
        self.TARGET_SIZE = 640

    def inferer(self, frames: list[VideoFrame]):
        req_logger = logger.bind(frames_count=len(frames))

        try:
            req_logger.info("Running batch inference on video frames")

            batches = [
                frames[i:i + self.BATCH_SIZE]
                for i in range(0, len(frames), self.BATCH_SIZE)
            ]

            for batch in batches:
                vectors = self.vectorizer.encode_batch_images(batch)

                for idx, frame in enumerate(batch):
                    description = self._get_image_description(frame)
                    
                    if vectors[idx].metadata is None:
                        vectors[idx].metadata = {}
                    
                    vectors[idx].metadata["description"] = description

                self.store.save_batch(vectors)
        except Exception:
            req_logger.exception("Failed to execute inference or save vector batches")

            raise

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

