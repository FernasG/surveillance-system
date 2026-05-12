import cv2
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.pipeline.preprocessing.mog2_frame_sampler import MOG2FrameSampler

def main():
    print("Exec")
    video = cv2.VideoCapture("/app/videos/video.mp4")
    sampler = MOG2FrameSampler()
    vectorizer = CLIPVectorizer()

    frames = sampler.get_frames(video)
    vectors = []

    for frame in frames:
        vector = vectorizer.encode_image(frame)
        vectors.append(vector)

    print(vectors[0])

    print("Finsihed")

if __name__ == "__main__":
    main()