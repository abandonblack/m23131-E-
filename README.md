# 猫狗品种精准识别平台（自训练版）

一个基于自定义网络训练的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示

## 推荐数据集（已内置在代码常量中）

训练脚本 `app/train.py` 内置 `DATASET_CATALOG`，当前包含：

1. **Oxford-IIIT Pet Dataset**（推荐首选）
   - 37 个猫狗品种，标注规范，适合该任务
   - 官网：<https://www.robots.ox.ac.uk/~vgg/data/pets/>
2. **Stanford Dogs Dataset**（狗品种增强）
   - 120 个犬种细粒度分类
   - 官网：<http://vision.stanford.edu/aditya86/ImageNetDogs/>
3. **Kaggle 猫品种数据集（补充项）**
   - 用于扩充猫品种样本，需注意标签清洗
   - 入口：<https://www.kaggle.com/datasets>

你也可以通过接口查看：`GET /api/datasets`。

## 网络结构说明

项目支持两种自定义结构：

- `breednet`：轻量 CNN，训练快，适合快速验证。
- `resbreednet`：残差网络（ResNet 思路实现），表达能力更强，通常精度更高，建议正式训练使用。

> 回答你的问题：是的，原始轻量网络可能偏简单；若要高精准，建议优先使用 `resbreednet`，并结合数据增强、类别均衡、较长训练周期。

## 1. 准备数据

将数据按 `ImageFolder` 结构放到 `data/breeds`：

```text
data/breeds/
  siamese_cat/
    xxx.jpg
  persian_cat/
    xxx.jpg
  golden_retriever/
    xxx.jpg
  ...
```

## 2. 训练模型

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.train --data-dir data/breeds --dataset-preset oxford_iiit_pet --arch resbreednet --epochs 20
```

训练后输出：

- `artifacts/breednet.pth`（权重）
- `artifacts/breednet.json`（训练信息）

## 3. 启动服务

```bash
python app/main.py
```

访问：`http://127.0.0.1:8000`

## 接口

- `GET /api/datasets`：查看推荐数据集清单
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
