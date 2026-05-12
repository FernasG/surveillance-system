import cv2

cap = cv2.VideoCapture("/app/videos/video.mp4")
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
fps = cap.get(cv2.CAP_PROP_FPS)

before_algo = 0
frames = []
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    count += 1

    if count % fps != 0:
        continue
    
    before_algo += 1
    mask = subtractor.apply(frame)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = False

    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            valid = True
            break
    
    if valid:
        frames.append(frame)

    # cv2.imshow("Filtro de Sombras", mask)
    # if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
# cv2.destroyAllWindows()

print(len(frames), before_algo)
