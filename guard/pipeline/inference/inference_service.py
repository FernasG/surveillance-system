from loguru import logger
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface
from guard.core.entities import VideoFrame

class InferenceService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface):
        self.vectorizer = vectorizer
        self.store = store
        self.BATCH_SIZE = 8

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

                self.store.save_batch(vectors)
        except Exception:
            req_logger.exception("Failed to execute inference or save vector batches")

            raise

