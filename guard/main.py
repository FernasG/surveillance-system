import cv2
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.pipeline.inference.inference_service import InferenceService
from guard.pipeline.preprocessing.preprocessor_service import PreprocessorService
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler

def main():
    print("Exec")
    video = cv2.VideoCapture("/app/videos/video.mp4")

    vectorizer = CLIPVectorizer()
    sampler = MOG2FrameSampler()
    store = ChromaDBStore()

    preprocessor_service = PreprocessorService(sampler=sampler)
    inferer_service = InferenceService(vectorizer=vectorizer, store=store)

    frames = preprocessor_service.process(video)
    inferer_service.inferer(frames)

    print("Finsihed")

if __name__ == "__main__":
    main()