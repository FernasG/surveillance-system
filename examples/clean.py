import cv2

cap = cv2.VideoCapture("/app/videos/video.mp4")

first_frame = None
frames = []
count = 0

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_pixels = height * width

before_algo = 0

SENSITIVITY_THRESHOLD = total_pixels * 0.005 * 255

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    count += 1

    if count % fps != 0:
        continue

    before_algo += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    if thresh.sum() > SENSITIVITY_THRESHOLD:
        frames.append(frame)

    first_frame = gray

print(len(frames), before_algo)
