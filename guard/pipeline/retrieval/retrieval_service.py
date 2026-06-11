import cv2
import base64
import numpy as np
from guard.infrastructure.models.utils.image_utils import cv2_to_base64
from PIL import Image
from loguru import logger
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, VLMInterface
from guard.core.entities import Settings, VLMMessage

class RetrievalService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface, vlm: VLMInterface, prompt_manager: PromptManager):
        self.settings = Settings()
        self.prompt_manager = prompt_manager
        self.vectorizer = vectorizer
        self.store = store
        self.vlm = vlm

    def search_by_text(self, text: str, top_k: int = 5):
        log_context = { "text": text, "top_k": top_k }
        req_logger = logger.bind(queue_message=log_context)

        try:
            req_logger.info("Starting RAG Query")

            query_vector = self.vectorizer.encode_text(text)
            search_result = self.store.search(query_vector, top_k=3)

            metadatas = search_result.get("metadatas", [])

            extracted_data = []

            if not metadatas or not metadatas[0]:
                return {"results": []}

            for metadata in metadatas[0]:
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

                extracted_data.append({
                    "video_path": video_path,
                    "elapsed_ms": elapsed_ms,
                    "raw_frame": frame
                })

            if not extracted_data:
                return {"results": []}
            
            frames = [data["raw_frame"] for data in extracted_data]
            messages, pil_images = self._setup_vlm_params(text, frames)

            response = self.vlm.generate(messages, pil_images, "json")

            final_results = []

            for eval_item in response.content:
                idx = eval_item.get("index")
                
                if idx is not None and 0 <= idx < len(extracted_data):
                    frame_context = extracted_data[idx]
                    
                    final_results.append({
                        "video_path": frame_context["video_path"],
                        "elapsed_ms": frame_context["elapsed_ms"],
                        "confidence_score": eval_item.get("confidence_score", 0.0),
                        "frame_base64": cv2_to_base64(frame_context["raw_frame"])
                    })

            final_results = sorted(final_results, key=lambda x: x["confidence_score"], reverse=True)

            return {"results": final_results}
        except Exception as e:
            req_logger.critical(f"Something went really wrong {e}")

            return {"error": str(e), "results": []}
        
    def _setup_vlm_params(self, text: str, frames: list[np.ndarray]) -> tuple[list[VLMMessage], list[Image.Image]]:
        pil_images: list[Image.Image] = []

        for frame in frames:
            small_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_images.append(pil_img)

        prompt_text = self.prompt_manager.build(
            prompt_name="search_evaluation",
            num_frames=len(frames),
            user_query=text
        )

        messages: list[VLMMessage] = [
            VLMMessage(role="user", content=prompt_text)
        ]

        return messages, pil_images
