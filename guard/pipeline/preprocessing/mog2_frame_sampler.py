import cv2
from guard.core.interfaces import VideoFrameSampler
from guard.core.entities import VideoFrame, QueueMessage

class MOG2FrameSampler(VideoFrameSampler):
    def __init__(self):
        super().__init__()

        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.DEFAULT_FPS = 30

    def get_frames(self, message: QueueMessage, video: cv2.VideoCapture) -> list[VideoFrame]:
        raw_fps = video.get(cv2.CAP_PROP_FPS)
        fps = int(round(raw_fps)) if raw_fps > 0 else self.DEFAULT_FPS
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

            motion_bbox = None

            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)

                    if motion_bbox is None:
                        motion_bbox = [x, y, x + w, y + h]
                    else:
                        motion_bbox[0] = min(motion_bbox[0], x)
                        motion_bbox[1] = min(motion_bbox[1], y)
                        motion_bbox[2] = max(motion_bbox[2], x + w)
                        motion_bbox[3] = max(motion_bbox[3], y + h)

            if motion_bbox is not None:
                video_frame = VideoFrame(
                    timestamp=message.timestamp,
                    frame_index=frame_count,
                    video_path=message.video_path,
                    elapsed_ms=int((frame_count / fps) * 1000),
                    motion_bbox=tuple(motion_bbox),
                    data=frame
                )

                frames.append(video_frame)

        video.release()

        return frames

