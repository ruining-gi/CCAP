import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from unet import Unet
from predata import LiverDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ====================== 配置参数 ======================
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
y_transforms = transforms.ToTensor()

train_image_dir = ""
train_mask_dir  = ""
val_image_dir   = ""
val_mask_dir    = ""

# ====================== 指标函数 ======================
def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    preds_flat = preds_bin.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    probs_flat = probs.view(-1).cpu().numpy()

    TP = np.sum((preds_flat == 1) & (targets_flat == 1))
    TN = np.sum((preds_flat == 0) & (targets_flat == 0))
    FP = np.sum((preds_flat == 1) & (targets_flat == 0))
    FN = np.sum((preds_flat == 0) & (targets_flat == 1))

    RC = TP / (TP + FN + eps)
    SP = TN / (TN + FP + eps)
    ACC = (TP + TN) / (TP + TN + FP + FN + eps)
    IOU = TP / (TP + FP + FN + eps)
    F1 = 2 * TP / (2 * TP + FP + FN + eps)

    try:
        AUC = roc_auc_score(targets_flat, probs_flat)
    except:
        AUC = np.nan

    return {
        "RC": RC * 100,
        "SP": SP * 100,
        "ACC": ACC * 100,
        "IOU": IOU * 100,
        "F1": F1 * 100,
        "AUC": AUC * 100
    }

# ====================== 训练函数 ======================
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 30)
        model.train()
        epoch_loss = 0

        for step, (x, y) in enumerate(train_loader):
            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

        # 验证阶段评估指标
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                val_out = model(val_x)
                preds_all.append(val_out)
                targets_all.append(val_y)

        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = compute_metrics(preds_all, targets_all)

        print("📊 Validation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}%")

        # 保存权重
        torch.save(model.state_dict(), f'weights_epoch{epoch + 1}.pth')

    return model

# ====================== 测试函数 ======================
def test():
    model = Unet(1, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    model = model.to(device)
    model.eval()

    val_dataset = LiverDataset(val_image_dir, val_mask_dir, transform=x_transforms, target_transform=y_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1)

    preds_all, targets_all = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            preds_all.append(out)
            targets_all.append(y)

            # 可视化
            prob = torch.sigmoid(out)
            img = torch.squeeze(prob).cpu().numpy()
            plt.imshow(img, cmap='gray')
            plt.pause(0.01)

        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = compute_metrics(preds_all, targets_all)

        print("===== Evaluation Metrics =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}%")

        plt.show()

# ====================== 训练封装 ======================
def train():
    model = Unet(1, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = LiverDataset(train_image_dir, train_mask_dir, transform=x_transforms, target_transform=y_transforms)
    val_dataset = LiverDataset(val_image_dir, val_mask_dir, transform=x_transforms, target_transform=y_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=args.epochs)

# ====================== 参数解析 ======================
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--ckp", type=str, default="weights_epoch20.pth", help="Path to model weights for testing")
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
args = parser.parse_args()

# ====================== 程序入口 ======================
if __name__ == "__main__":
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
