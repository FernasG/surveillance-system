import cv2
from guard.core.interfaces import VideoFrameSampler
from guard.core.entities import VideoFrame, QueueMessage

class MOG2FrameSampler(VideoFrameSampler):
    def __init__(self):
        super().__init__()

        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def get_frames(self, message: QueueMessage, video: cv2.VideoCapture) -> list[VideoFrame]:
        fps = video.get(cv2.CAP_PROP_FPS)
        frames: list[VideoFrame] = []

        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
    
            if not ret:
                break

            frame_count += 1

            if frame_count % fps != 0:
                continue

            mask = self.subtractor.apply(frame)

            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid = False

            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:
                    valid = True
                    break

            if valid:
                video_frame = VideoFrame(
                    timestamp=message.timestamp,
                    frame_index=frame_count,
                    video_path=message.video_path,
                    elapsed_ms=int((frame_count / fps) * 1000),
                    data=frame
                )

                frames.append(video_frame)

        video.release()

        return frames

