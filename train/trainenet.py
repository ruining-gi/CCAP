import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from dataset import LiverDataset
from enet_original import ENet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ====================== 配置参数 ======================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 数据预处理（单通道转三通道适配ENet）
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 单通道归一化
])
y_transforms = transforms.ToTensor()

# 数据集路径
train_image_dir = "/mnt/diskb5/ruining_private/-SylveStar-/SGTA1500use/train/image"
train_mask_dir = "/mnt/diskb5/ruining_private/-SylveStar-/SGTA1500use/train/mask"
val_image_dir = "/mnt/diskb5/ruining_private/-SylveStar-/SGTA1500use/val/image"
val_mask_dir = "/mnt/diskb5/ruining_private/-SylveStar-/SGTA1500use/val/mask"
test_image_dir = "/mnt/diskb5/ruining_private/data2000test/img"
test_mask_dir = "/mnt/diskb5/ruining_private/data2000test/datapre"

# 结果保存路径
save_vis_dir = "./result/"
os.makedirs(save_vis_dir, exist_ok=True)

# ====================== 损失函数 ======================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, eps=1e-7, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        
    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss

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
    IOU = TP / (TP + FP + FN + eps)  # 关键指标：IOU
    F1 = 2 * TP / (2 * TP + FP + FN + eps)

    try:
        AUC = roc_auc_score(targets_flat, probs_flat)
    except:
        AUC = np.nan

    return {"RC": RC*100, "SP": SP*100, "ACC": ACC*100, "IOU": IOU*100, "F1": F1*100, "AUC": AUC*100}

# ====================== 训练函数（修改为IOU最佳保存） ======================
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, best_path):
    best_val_iou = -float("inf")  # 初始化最佳IOU为负无穷（IOU越大越好）
    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 50)
        model.train()
        epoch_loss = 0

        # 训练阶段
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.repeat(1, 3, 1, 1)  # 单通道转三通道
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            current_batch_loss = loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Batch [{batch_idx + 1}/{total_batches}] - Loss: {current_batch_loss:.4f}")

        avg_loss = epoch_loss / total_batches
        print(f"\nEpoch {epoch + 1} 训练完成 - 平均损失: {avg_loss:.4f}")

        # 验证阶段
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.repeat(1, 3, 1, 1)
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_out = model(val_x)
                preds_all.append(val_out)
                targets_all.append(val_y)

        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = compute_metrics(preds_all, targets_all)  # 计算包括IOU在内的指标

        print("\U0001F4CA Validation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}%")

        # 核心修改：以IOU作为最佳模型判断依据
        current_val_iou = metrics["IOU"]  # 获取当前验证集IOU
        if current_val_iou > best_val_iou:  # 若当前IOU高于历史最佳
            best_val_iou = current_val_iou  # 更新最佳IOU
            torch.save(model.state_dict(), best_path)  # 保存模型
            print(f"✅ 保存最佳模型 (当前IOU: {current_val_iou:.2f}%，历史最佳: {best_val_iou:.2f}%)")
        else:
            print(f"⏸️ 未保存模型 (当前IOU: {current_val_iou:.2f}%，历史最佳: {best_val_iou:.2f}%)")

    return model

# ====================== 测试函数 ======================
def test_model(model, test_loader):
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.repeat(1, 3, 1, 1)
            x, y = x.to(device), y.to(device)
            out = model(x)

            preds_all.append(out)
            targets_all.append(y)

            # 生成预测掩码
            prob = torch.sigmoid(out)
            pred_mask = (prob > 0.5).float()

            # 可视化并保存
            plt.figure(figsize=(12, 4))
            plt.suptitle(f"Test Sample {idx + 1}/{len(test_loader)}")
            plt.subplot(131)
            plt.title("Original Image")
            plt.imshow(torch.squeeze(x[:, 0, :, :]).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(132)
            plt.title("True Mask")
            plt.imshow(torch.squeeze(y).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(133)
            plt.title("Predicted Mask")
            plt.imshow(torch.squeeze(pred_mask).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            
            save_path = os.path.join(save_vis_dir, f"test_sample_{idx}.png")
            plt.savefig(save_path)
            print(f"Saved test result: {save_path}")
            plt.close()

    # 计算test集指标
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    metrics = compute_metrics(preds_all, targets_all)

    print("\n===== Test Set Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")

# ====================== 主函数 ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckp", type=str, default="best_model.pth", help="测试时加载的模型路径")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--loss_type", type=str, default="dice", choices=["dice", "bce", "dice_bce"])
    args = parser.parse_args()

    # 最佳模型保存路径
    best_path = "/mnt/diskb5/ruining_private/20250725SylveL/enet/best_model.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    # 加载数据集
    train_dataset = LiverDataset(train_image_dir, train_mask_dir, transform=x_transforms, target_transform=y_transforms)
    val_dataset = LiverDataset(val_image_dir, val_mask_dir, transform=x_transforms, target_transform=y_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test_dataset = LiverDataset(test_image_dir, test_mask_dir, transform=x_transforms, target_transform=y_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 初始化模型
    model = ENet(n_classes=1).to(device)

    if args.mode == "train":
        # 选择损失函数
        if args.loss_type == "dice":
            criterion = DiceLoss().to(device)
        elif args.loss_type == "bce":
            criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            criterion = DiceBCELoss(weight_dice=0.7, weight_bce=0.3).to(device)
            
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, args.epochs, best_path)
        print("\n=======================")
        print("Training finished. Starting test on test set...")
        test_model(trained_model, test_loader)
    
    elif args.mode == "test":
        model.load_state_dict(torch.load(args.ckp, map_location=device))
        test_model(model, test_loader)

if __name__ == "__main__":
    main()
