# 猫狗品种精准识别平台（Oxford-IIIT + ResNet34-SE 版）

一个基于 **PyTorch + FastAPI** 的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示

## 当前训练配置（针对精度优化）

项目保留 ResNet 路线，并针对你反馈的低准确率做了强化：

- 在 `model.py` 中显式实现 ResNet34-SE 各层（不再直接调用模型接口）
- 数据增强：`RandomResizedCrop` / 翻转 / 颜色扰动 / `RandomErasing`
- 优化器升级为 `AdamW` + `CosineAnnealingLR`
- 损失函数支持 `label_smoothing`
- 分层划分训练/验证集，避免类别分布偏差

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 训练模型（默认自动下载 Oxford-IIIT）

```bash
python -m app.train --data-dir data/oxford_iiit_pet --epochs 40
```

常用参数：

- `--no-download`：不自动下载
- `--batch-size 24 --lr 5e-4 --weight-decay 5e-4 --image-size 256`

训练后输出：

- `artifacts/breednet.pth`（权重）
- `artifacts/breednet.json`（训练信息）

## 3. 启动服务

```bash
python app/main.py
```

访问：`http://127.0.0.1:8000`

## 接口

- `GET /api/datasets`：查看数据集清单（当前仅 Oxford-IIIT）
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
