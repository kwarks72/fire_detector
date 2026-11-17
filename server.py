from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="fire-detector server")

Path("captures").mkdir(exist_ok=True)
app.mount("/captures", StaticFiles(directory="captures"), name="captures")

events = []  # 감지 기록 (서버를 끄면 사라져요)


@app.post("/events")
async def add_event(
    label: str = Form(...),
    confidence: float = Form(...),
    image: UploadFile = File(...),
):
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    (Path("captures") / filename).write_bytes(await image.read())

    event = {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "confidence": round(confidence, 2),
        "image": f"/captures/{filename}",
    }
    events.insert(0, event)
    return {"ok": True}


@app.get("/events")
def list_events():
    return events


@app.get("/", response_class=HTMLResponse)
def dashboard():
    rows = "".join(
        f"<tr><td>{e['time']}</td><td>{e['label']}</td><td>{e['confidence']}</td>"
        f"<td><img src='{e['image']}' width='200'></td></tr>"
        for e in events
    )
    return f"""<html><head><meta charset='utf-8'>
    <meta http-equiv='refresh' content='3'><title>화재 감지 대시보드</title></head>
    <body><h1>🔥 화재·연기 감지 기록</h1>
    <p>총 {len(events)}건 (3초마다 자동 새로고침)</p>
    <table border='1' cellpadding='6'>
    <tr><th>시간</th><th>종류</th><th>확신도</th><th>캡처</th></tr>{rows}</table>
    </body></html>"""