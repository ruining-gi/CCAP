import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import LiverDataset
from deeplabv3_2 import DeepLab  
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 基础transform，无数据增强
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize([0.5], [0.5])
])

y_transforms = transforms.ToTensor()

# 路径配置
train_image_dir = "/mnt/diskb5/ruining_private/CHENXI/adjust/SGATdata/train/image/"
train_mask_dir  = "/mnt/diskb5/ruining_private/CHENXI/adjust/SGATdata/train/mask/"
val_image_dir   = "/mnt/diskb5/ruining_private/CHENXI/adjust/SGATdata/val/image/"
val_mask_dir    = "/mnt/diskb5/ruining_private/CHENXI/adjust/SGATdata/val/mask/"
test_image_dir  = "/mnt/diskb5/ruining_private/CHENXI/adjust/SGATdata/test/image/"
test_mask_dir   = "/mnt/diskb5/ruining_private/CHENXI/adjust/SGATdata/test/mask/" 

save_vis_dir = "/mnt/diskb5/ruining_private/CHENXI/adjust/DeepLabV3/result4/deeplab_visualizations/"
os.makedirs(save_vis_dir, exist_ok=True)

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

    return {"RC": RC*100, "SP": SP*100, "ACC": ACC*100, "IOU": IOU*100, "F1": F1*100, "AUC": AUC*100}

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, scheduler=None):
    best_val_loss = float("inf")
    best_path = f"/mnt/diskb5/ruining_private/CHENXI/adjust/weight/DeepLab4/best_model.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    
    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(x)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"训练损失: {avg_loss:.4f}")

        # 验证阶段
        model.eval()
        preds_all, targets_all = [], []
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_out = model(val_x)
                # 根据损失函数类型决定是否使用sigmoid
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    loss = criterion(val_out, val_y)
                else:
                    loss = criterion(torch.sigmoid(val_out), val_y)
                val_loss += loss.item()
                preds_all.append(val_out)
                targets_all.append(val_y)

        avg_val_loss = val_loss / len(val_loader)
        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = compute_metrics(preds_all, targets_all)

        print("📊 验证指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}%")
        print(f"验证损失: {avg_val_loss:.4f}")
        
        if scheduler is not None:
            if isinstance(scheduler, (CosineAnnealingLR, OneCycleLR)):
                scheduler.step()
            else:
                scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            print("✅ 保存最佳模型")

    return model

# Dice Loss 实现
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 添加sigmoid确保输入在0-1范围内
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

def train(model_class):
    model = model_class(num_classes=1, in_channels=1).to(device)
    
    # 可选择优化器
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # 可选择学习率调度器
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs)
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=args.lr/100)
    
    # 修改损失函数选择部分
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()  # 替换为安全的WithLogits版本
    elif args.loss == "dice":
        criterion = DiceLoss()
    else:  # combined
        bce_loss = nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss()
        criterion = lambda pred, target: 0.5*bce_loss(pred, target) + 0.5*dice_loss(pred, target)

    train_dataset = LiverDataset(train_image_dir, train_mask_dir, 
                               transform=x_transforms, target_transform=y_transforms)
    val_dataset = LiverDataset(val_image_dir, val_mask_dir,
                             transform=x_transforms, target_transform=y_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    train_model(model, criterion, optimizer, train_loader, val_loader, 
               num_epochs=args.epochs, scheduler=scheduler)

def test(model_class):
    model = model_class(num_classes=1, in_channels=1, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location=device))
    model.eval()

    test_dataset = LiverDataset(test_image_dir, test_mask_dir, transform=x_transforms, target_transform=y_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)

    preds_all, targets_all = [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            preds_all.append(out)
            targets_all.append(y)

            prob = torch.sigmoid(out)
            pred_mask = (prob > args.threshold).float()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(torch.squeeze(x).cpu().numpy(), cmap='gray')
            axs[0].set_title('Input Image'); axs[0].axis('off')
            axs[1].imshow(torch.squeeze(pred_mask).cpu().numpy(), cmap='gray')
            axs[1].set_title('Predicted Mask'); axs[1].axis('off')
            axs[2].imshow(torch.squeeze(y).cpu().numpy(), cmap='gray')
            axs[2].set_title('Ground Truth Mask'); axs[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_vis_dir, f"sample_{idx}.png")
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    metrics = compute_metrics(preds_all, targets_all, threshold=args.threshold)

    print("===== Evaluation Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (1e-4 to 5e-4 recommended)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], 
                       help="Optimizer choice")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine", "onecycle"], 
                       help="Learning rate scheduler")
    parser.add_argument("--loss", type=str, default="combined", choices=["bce", "dice", "combined"], 
                       help="Loss function")
    
    # 测试参数
    parser.add_argument("--ckp", type=str, 
                       default="/mnt/diskb5/ruining_private/CHENXI/adjust/weight/DeepLab4/best_model.pth", 
                       help="Path to model weights for testing")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Threshold for binary prediction in testing")
    parser.add_argument("--backbone", type=str, default="xception", 
                       choices=["xception", "resnet101", "resnet50"], 
                       help="Backbone model to use (xception or resnet系列)")
    args = parser.parse_args()

    model_dict = {
        "DeepLab": DeepLab
    }

    if args.mode == "train":
        train(model_dict["DeepLab"])
    elif args.mode == "test":
        test(model_dict["DeepLab"])