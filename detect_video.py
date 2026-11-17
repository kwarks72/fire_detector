import time

import cv2
import requests
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

SERVER_URL = "http://127.0.0.1:8000/events"
COOLDOWN = 5          # 신고 사이의 최소 간격(초)
SOURCE = "fire.mp4"   # 웹캠을 쓰려면 0

path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
model = YOLO(path)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("영상을 열 수 없어요. 파일 이름과 위치를 확인하세요:", SOURCE)
    raise SystemExit

last_sent = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    boxes = results[0].boxes

    if len(boxes) > 0 and time.time() - last_sent > COOLDOWN:
        best = boxes[int(boxes.conf.argmax())]
        label = model.names[int(best.cls)]
        conf = float(best.conf)

        ok_jpg, buf = cv2.imencode(".jpg", annotated)
        try:
            requests.post(
                SERVER_URL,
                data={"label": label, "confidence": conf},
                files={"image": ("capture.jpg", buf.tobytes(), "image/jpeg")},
                timeout=3,
            )
            print(f"알림 전송: {label} {conf:.2f}")
        except requests.RequestException:
            print("서버에 연결할 수 없어요. 서버가 켜져 있나요?")
        last_sent = time.time()

    cv2.imshow("fire-detector (q: quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()