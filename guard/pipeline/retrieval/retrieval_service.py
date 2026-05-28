import cv2, os
import numpy as np
from PIL import Image
from loguru import logger
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, VLMInterface
from guard.core.entities import Settings, VLMMessage

class RetrievalService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface, vlm: VLMInterface):
        self.settings = Settings()
        self.vectorizer = vectorizer
        self.store = store
        self.vlm = vlm

    def search_by_text(self, text: str, top_k: int = 5):
        log_context = {
            "text": text,
            "top_k": top_k
        }
        req_logger = logger.bind(queue_message=log_context)

        try:
            req_logger.info("Starting RAG Query")

            query_vector = self.vectorizer.encode_text(text)
            search_result = self.store.search(query_vector, top_k=top_k)

            metadatas = search_result.get("metadatas", [])
            frames: list[np.ndarray] = []

            if not metadatas or not metadatas[0]:
                return {"frames": frames}
            
            metadatas = metadatas[0]

            for metadata in metadatas:
                elapsed_ms = metadata.get("elapsed_ms")
                video_path = metadata.get("video_path")

                if not elapsed_ms or not video_path:
                    continue

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    continue

                cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)

                success, frame = cap.read()

                cap.release()

                if not success or frame is None:
                    continue

                frames.append(frame)

            if not frames:
                return {"frames": frames}
            
            messages, pil_images = self._setup_vlm_params(text, frames)

            response_text = self.vlm.generate(messages, pil_images)

            return {"response_text": response_text}
        except Exception as e:
            req_logger.critical(f"Something went really wrong {e}")

        
    def _setup_vlm_params(self, text: str, frames: list[np.ndarray]) -> tuple[list[VLMMessage], list[Image.Image]]:
        pil_images: list[Image.Image] = []
        content_list = []

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_images.append(pil_img)
            
            content_list.append({"type": "image"})

        content_list.append({"type": "text", "text": f"Order the images that most represents user query: '{text}'"})

        messages: list[VLMMessage] = [
            VLMMessage(role="user", content=content_list)
        ]

        return messages, pil_images
                    