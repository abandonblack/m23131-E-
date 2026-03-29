from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from app.model import build_model

DATASET_CATALOG = {
    "oxford_iiit_pet": {
        "name": "Oxford-IIIT Pet Dataset",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "description": "37 个猫狗品种，官方标注质量高，适合品种分类。",
    },
    "stanford_dogs": {
        "name": "Stanford Dogs Dataset",
        "url": "http://vision.stanford.edu/aditya86/ImageNetDogs/",
        "description": "120 个犬种细粒度分类，适合狗品种识别强化。",
    },
    "cat_breeds_dataset": {
        "name": "Cat Breeds Dataset (Kaggle 社区常见版本)",
        "url": "https://www.kaggle.com/datasets",
        "description": "可补充猫品种样本，但需注意标签清洗与授权协议。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练猫狗品种分类模型（自定义网络）")
    parser.add_argument("--data-dir", type=Path, default=Path("data/breeds"))
    parser.add_argument("--dataset-preset", type=str, default="oxford_iiit_pet", choices=list(DATASET_CATALOG.keys()))
    parser.add_argument("--arch", type=str, default="resbreednet", choices=["breednet", "resbreednet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/breednet.pth"))
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    dataset_info = DATASET_CATALOG[args.dataset_preset]
    print(f"数据集建议：{dataset_info['name']} | {dataset_info['url']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=str(args.data_dir), transform=transform)
    if len(dataset.classes) < 2:
        raise ValueError("至少需要两个品种类别进行训练")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.arch, num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_set), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "arch": args.arch,
                    "model_state_dict": model.state_dict(),
                    "classes": dataset.classes,
                    "image_size": args.image_size,
                    "dataset_preset": args.dataset_preset,
                },
                args.save_path,
            )

    meta_path = args.save_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "arch": args.arch,
                "dataset_preset": args.dataset_preset,
                "dataset_info": dataset_info,
                "classes": dataset.classes,
                "best_val_acc": best_acc,
                "epochs": args.epochs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"权重已保存: {args.save_path}")


if __name__ == "__main__":
    main()
