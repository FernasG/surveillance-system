import cv2

cap = cv2.VideoCapture("/app/videos/video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps if fps > 0 else 0

print(f"FPS: {fps}")
print(f"Resolution: {int(width)}x{int(height)}")
print(f"Frame count: {int(frame_count)}")
print(f"Duration (seconds): {duration:.2f}")

cap.release()