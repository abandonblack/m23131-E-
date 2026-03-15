from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FEEDBACK_FILE = BASE_DIR / "feedback.json"

UPLOAD_DIR.mkdir(exist_ok=True)
if not FEEDBACK_FILE.exists():
    FEEDBACK_FILE.write_text("[]", encoding="utf-8")

app = FastAPI(title="猫狗品种智能识别")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

CLASSIFIER = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
)

BREED_LABELS = [
    "Siamese cat",
    "Persian cat",
    "Maine Coon cat",
    "Ragdoll cat",
    "British Shorthair cat",
    "Bengal cat",
    "Sphynx cat",
    "Golden Retriever dog",
    "Labrador Retriever dog",
    "German Shepherd dog",
    "Poodle dog",
    "French Bulldog dog",
    "Siberian Husky dog",
    "Shiba Inu dog",
]

BREED_INFO = {
    "Siamese cat": "暹罗猫性格外向、互动性强，叫声辨识度高，适合陪伴型家庭。",
    "Persian cat": "波斯猫被毛浓密、气质温和，需要定期梳毛和眼部清洁。",
    "Maine Coon cat": "缅因猫体型大、友好聪明，是非常受欢迎的大型长毛猫。",
    "Ragdoll cat": "布偶猫性格温顺、黏人，常被称为“行走的棉花糖”。",
    "British Shorthair cat": "英短结实圆润、适应能力强，适合城市家庭饲养。",
    "Bengal cat": "孟加拉猫运动需求高、花纹醒目，需要较多环境刺激。",
    "Sphynx cat": "斯芬克斯无毛猫亲人活泼，对温度较敏感，需要注重保暖。",
    "Golden Retriever dog": "金毛犬温顺友善、服从性高，是经典家庭伴侣犬。",
    "Labrador Retriever dog": "拉布拉多智商高、精力旺盛，适合喜欢户外活动的家庭。",
    "German Shepherd dog": "德国牧羊犬忠诚警觉，训练潜力高，常见于工作犬领域。",
    "Poodle dog": "贵宾犬聪明易训、掉毛少，按体型分为多种品系。",
    "French Bulldog dog": "法国斗牛犬体型小、性格稳定，但需注意呼吸道健康。",
    "Siberian Husky dog": "哈士奇精力充沛、耐力强，需要充分运动和社交。",
    "Shiba Inu dog": "柴犬独立机警、表情丰富，训练时需耐心和一致性。",
}


def predict_breed(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path).convert("RGB") as image:
        predictions = CLASSIFIER(image, candidate_labels=BREED_LABELS)

    best = max(predictions, key=lambda x: x["score"])
    return {
        "label": best["label"],
        "confidence": round(float(best["score"]) * 100, 2),
        "description": BREED_INFO.get(best["label"], "暂无该品种详细介绍。"),
        "top3": sorted(predictions, key=lambda x: x["score"], reverse=True)[:3],
    }


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    suffix = Path(image.filename or "").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    file_path = UPLOAD_DIR / filename

    content = await image.read()
    file_path.write_bytes(content)

    result = predict_breed(file_path)
    result["image_url"] = f"/uploads/{filename}"
    return JSONResponse(result)


@app.post("/api/feedback")
def submit_feedback(
    nickname: str = Form(...),
    message: str = Form(...),
    rating: int = Form(...),
):
    try:
        feedback_list = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        feedback_list = []

    feedback = {
        "nickname": nickname,
        "message": message,
        "rating": rating,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    feedback_list.append(feedback)
    FEEDBACK_FILE.write_text(
        json.dumps(feedback_list[-100:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"message": "反馈提交成功，感谢你的建议！"}


@app.get("/api/feedback")
def list_feedback():
    try:
        feedback_list = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        feedback_list = []
    return {"items": list(reversed(feedback_list[-10:]))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
