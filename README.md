# 猫狗品种精准识别平台

一个基于深度学习模型（CLIP）的 Web 应用，支持：

- 图片上传并识别猫狗常见品种
- 自动展示识别置信度与品种介绍
- 用户反馈提交与最近反馈展示

## 技术栈

- FastAPI + Jinja2
- Transformers `openai/clip-vit-base-patch32`
- 原生 HTML/CSS/JS

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

启动后访问：`http://127.0.0.1:8000`

## 接口

- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：表单字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
