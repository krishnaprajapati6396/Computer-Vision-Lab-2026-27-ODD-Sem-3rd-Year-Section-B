import cv2
import os
import time
from datetime import datetime
from config import CAMERA_URL

os.makedirs("snapshots", exist_ok=True)

MIN_AREA = 1500          # isse bada moving object = motion (kam karo to sensitive)
SNAPSHOT_GAP = 3         # seconds, itne gap par hi nayi photo save hogi

cap = cv2.VideoCapture(CAMERA_URL)
ret, frame = cap.read()
if not ret:
    print("Camera connect nahi hua. IP aur Wi-Fi check karo.")
    exit()

def prepare(img):
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, cv2.GaussianBlur(gray, (21, 21), 0)

frame, prev_gray = prepare(frame)
last_saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame, gray = prepare(frame)

    # 1. Frame subtraction
    diff = cv2.absdiff(prev_gray, gray)
    # 2. Threshold + dilate
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    # 3. Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    motion = False
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        motion = True
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion:
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: MOTION DETECTED!", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # snapshot save (har frame par nahi, gap ke saath)
        if time.time() - last_saved > SNAPSHOT_GAP:
            name = datetime.now().strftime("snapshots/alert_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(name, frame)
            print("Saved:", name)
            last_saved = time.time()
    else:
        cv2.putText(frame, "Status: Clear", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    prev_gray = gray
    cv2.imshow("Security Camera", frame)
    cv2.imshow("Threshold (debug)", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()