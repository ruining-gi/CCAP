import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import LiverDataset
from sk import SKNetSeg
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
# 新增：导入PIL用于单独保存分割结果
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 基础transform，无数据增强
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize([0.5], [0.5])
])

y_transforms = transforms.ToTensor()

# 路径配置
train_image_dir = ""
train_mask_dir  = ""
val_image_dir   = ""
val_mask_dir    = ""
test_image_dir  = ""
test_mask_dir   = "" 

save_vis_dir = ""
# 新增：分割结果单独存储的文件夹路径
save_seg_dir = ""
os.makedirs(save_vis_dir, exist_ok=True)
# 新增：创建分割结果文件夹
os.makedirs(save_seg_dir, exist_ok=True)

# -------------------------- 1. 核心修改：compute_metrics 返回逐样本指标 --------------------------
def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    
    # 保留批次维度（batch_size），仅展平空间维度（H*W），确保每个样本独立计算指标
    batch_size = preds.shape[0]
    preds_flat = preds_bin.view(batch_size, -1).cpu().numpy()  # 形状：[batch_size, H*W]
    targets_flat = targets.view(batch_size, -1).cpu().numpy()  # 形状：[batch_size, H*W]
    probs_flat = probs.view(batch_size, -1).cpu().numpy()      # 形状：[batch_size, H*W]

    # 逐样本计算 TP/TN/FP/FN（按空间维度求和，得到每个样本的统计值）
    TP = np.sum((preds_flat == 1) & (targets_flat == 1), axis=1)  # 形状：[batch_size]
    TN = np.sum((preds_flat == 0) & (targets_flat == 0), axis=1)
    FP = np.sum((preds_flat == 1) & (targets_flat == 0), axis=1)
    FN = np.sum((preds_flat == 0) & (targets_flat == 1), axis=1)

    # 逐样本计算核心指标（避免除零错误）
    RC = TP / (TP + FN + eps)    # 召回率
    SP = TN / (TN + FP + eps)    # 特异度
    ACC = (TP + TN) / (TP + TN + FP + FN + eps)  # 准确率
    IOU = TP / (TP + FP + FN + eps)  # 交并比
    F1 = 2 * TP / (2 * TP + FP + FN + eps)  # F1分数

    # 逐样本计算AUC（处理标签全0/全1的异常情况，用NaN标记无效值）
    AUC = []
    for i in range(batch_size):
        try:
            auc = roc_auc_score(targets_flat[i], probs_flat[i])
        except:  # 标签全0或全1时无法计算AUC
            auc = np.nan
        AUC.append(auc)
    AUC = np.array(AUC)  # 形状：[batch_size]

    # 返回逐样本指标（转为百分比，便于阅读）
    return {
        "RC": RC * 100,
        "SP": SP * 100,
        "ACC": ACC * 100,
        "IOU": IOU * 100,
        "F1": F1 * 100,
        "AUC": AUC * 100
    }

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, scheduler=None):
    best_val_loss = float("inf")
    best_path = "/mnt/diskb5/ruining_private/CHENXI/adjust/weight/SK_ARCADE/best_model.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    
    scaler = torch.cuda.amp.GradScaler()  

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"训练损失: {avg_loss:.4f}")

        # 验证阶段：适配逐样本指标，用nanmean取均值（忽略AUC的NaN值）
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
            print(f"  {k}: {np.nanmean(v):.2f}%")  # 用nanmean过滤无效的NaN值
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

# Dice Loss 实现（不变）
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

# Combo Loss 实现（不变）
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, gamma=2.0, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        # Dice Loss部分
        prob = torch.sigmoid(pred)
        intersection = (prob * target).sum()
        dice = (2. * intersection + self.smooth) / (prob.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice
        
        # Focal Loss部分
        bce = self.bce(pred, target)
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma * bce).mean()
        
        # 组合损失
        return self.alpha * dice_loss + (1 - self.alpha) * (self.ce_ratio * focal_loss + (1 - self.ce_ratio) * bce.mean())

# 训练函数（不变）
def train():
    model = SKNetSeg(in_channels=1).to(device)
    
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
    
    # 损失函数选择（不变）
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "combo":
        criterion = ComboLoss(alpha=0.7, ce_ratio=0.5, gamma=2.0)
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

# -------------------------- 2. 核心修改：test 函数收集指标并计算 mean±std --------------------------
def test():
    # 初始化模型：添加weights_only=True消除PyTorch安全警告
    model = SKNetSeg(in_channels=1).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location=device, weights_only=True))
    model.eval()

    test_dataset = LiverDataset(test_image_dir, test_mask_dir, transform=x_transforms, target_transform=y_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)  # 批次1，便于逐样本收集指标

    # 新增：收集所有样本的指标（每个指标对应一个列表）
    all_metrics = {
        "RC": [], "SP": [], "ACC": [], "IOU": [], "F1": [], "AUC": []
    }

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            # 1. 计算当前样本的指标（batch_size=1，返回长度为1的数组）
            batch_metrics = compute_metrics(out, y, threshold=args.threshold)
            
            # 2. 将当前样本的指标加入全局列表（取数组第0个元素，即当前样本值）
            for metric_name, values in batch_metrics.items():
                all_metrics[metric_name].append(values[0])

            # 3. 可视化逻辑（不变）
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
            print(f"Saved visualization: {save_path}")
            plt.close()

            # 新增：单独保存分割结果（预测掩码）
            # 将tensor转为0-255的numpy数组（便于保存为图像）
            seg_mask = torch.squeeze(pred_mask).cpu().numpy()  # 去除batch和channel维度
            seg_mask = (seg_mask * 255).astype(np.uint8)      # 0->0, 1->255
            # 保存为PNG图像（无损格式）
            seg_save_path = os.path.join(save_seg_dir, f"segmentation_{idx}.png")
            seg_image = Image.fromarray(seg_mask, mode='L')   # 'L'表示灰度图
            seg_image.save(seg_save_path)
            print(f"Saved segmentation: {seg_save_path}")

    # 4. 计算所有样本的均值和标准差（用nanmean/nanstd忽略AUC的NaN值）
    final_results = {}
    for metric_name, values in all_metrics.items():
        values_np = np.array(values)
        mean_val = np.nanmean(values_np)  # 忽略无效的NaN值
        std_val = np.nanstd(values_np)    # 忽略无效的NaN值
        final_results[metric_name] = (mean_val, std_val)

    # 5. 打印结果：格式为「指标名: 均值% ± 标准差%」
    print("\n===== Test Evaluation Metrics (Mean ± Std) =====")
    for metric_name, (mean_val, std_val) in final_results.items():
        print(f"{metric_name}: {mean_val:.2f}% ± {std_val:.2f}%")

# 主函数（不变）
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckp", type=str, default="/mnt/diskb5/ruining_private/CHENXI/adjust/weight/SK_ARCADE/best_model.pth", 
                       help="Path to model weights for testing")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (1e-4 to 5e-4 recommended)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], 
                       help="Optimizer choice")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine", "onecycle"], 
                       help="Learning rate scheduler")
    parser.add_argument("--loss", type=str, default="combo", choices=["combo","bce", "dice", "combined"], 
                       help="Loss function")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Threshold for binary prediction in testing")
    
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
