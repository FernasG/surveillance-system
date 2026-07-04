import cv2
import base64
import requests
from loguru import logger
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface
from guard.core.entities import VideoFrame

class InferenceService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface):
        self.vectorizer = vectorizer
        self.store = store
        self.BATCH_SIZE = 8

        # self.qwen_url = "http://localhost:8081/v1/chat/completions"
        self.qwen_url = "http://192.168.1.10:8081/v1/chat/completions"

    def _get_image_description(self, frame: VideoFrame) -> str:
        try:
            success, buffer = cv2.imencode('.jpg', frame.data)
            
            if not success:
                raise ValueError("Could not encode frame to JPEG format")

            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "model": "qwen",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this video frame image in one simple sentence."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 80
            }

            response = requests.post(self.qwen_url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Failed to generate description for frame: {e}")
            return "Description unavailable"

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

