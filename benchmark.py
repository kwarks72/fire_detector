import time

import cv2
import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
pt_model = YOLO(path)

onnx_path = pt_model.export(format="onnx", imgsz=640)
onnx_model = YOLO(onnx_path)

img = cv2.imread("test_images/fire2.jpg")


def bench(model, imgsz=640, n=50):
    for _ in range(5):  # 워밍업
        model(img, imgsz=imgsz, device="cpu", verbose=False)
    start = time.perf_counter()
    for _ in range(n):
        model(img, imgsz=imgsz, device="cpu", verbose=False)
    sec = (time.perf_counter() - start) / n
    return 1 / sec  # FPS


cases = [
    ("PyTorch 640", pt_model, 640),
    ("PyTorch 480", pt_model, 480),
    ("PyTorch 320", pt_model, 320),
    ("ONNX 640", onnx_model, 640),
]

names, fps_list = [], []
for name, model, size in cases:
    fps = bench(model, size)
    print(f"{name}: {1000 / fps:.1f} ms/장, {fps:.1f} FPS")
    names.append(name)
    fps_list.append(fps)

plt.figure(figsize=(7, 4))
plt.bar(names, fps_list)
for i, v in enumerate(fps_list):
    plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
plt.ylabel("FPS (higher is faster)")
plt.title("Inference speed on CPU")
plt.tight_layout()
plt.savefig("benchmark.png", dpi=150)
print("benchmark.png 저장 완료")