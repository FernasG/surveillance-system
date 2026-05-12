from guard.core.interfaces import VectorizerInterface, VectorStoreInterface
from guard.core.entities import VideoFrame

class InferenceService():
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface):
        self.vectorizer = vectorizer
        self.store = store
        self.BATCH_SIZE = 8

    def inferer(self, frames: list[VideoFrame]):
        batches = [
            frames[i:i + self.BATCH_SIZE]
            for i in range(0, len(frames), self.BATCH_SIZE)
        ]

        for batch in batches:
            vectors = self.vectorizer.encode_batch_images(batch)

            self.store.save_batch(vectors, doc_id="strdoc1")
