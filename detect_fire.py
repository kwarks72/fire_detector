from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
model = YOLO(path)

results = model("test_images", conf=0.1, save=True)

for r in results:
    found = [(model.names[int(b.cls)], round(float(b.conf), 2)) for b in r.boxes]
    print(r.path, "->", found)