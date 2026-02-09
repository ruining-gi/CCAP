import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32, dropout_rate=0.1):
        super(SKConv, self).__init__()
        d = int(features / r)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False),
                nn.Dropout2d(p=dropout_rate)  # Added dropout after each branch
            ) for i in range(M)
        ])
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([nn.Linear(d, features) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout for attention computation

    def forward(self, x):
        batch_size = x.size(0)
        feas = torch.stack([conv(x) for conv in self.convs], dim=1)  # [B, M, C, H, W]
        fea_U = torch.sum(feas, dim=1)  # [B, C, H, W]
        fea_s = fea_U.mean(-1).mean(-1)  # GAP -> [B, C]
        fea_s = self.dropout(fea_s)  # Apply dropout to GAP features
        fea_z = self.fc(fea_s)  # [B, d]
        attention_vectors = torch.stack([fc(fea_z) for fc in self.fcs], dim=1)  # [B, M, C]
        attention_vectors = self.softmax(attention_vectors).unsqueeze(-1).unsqueeze(-1)  # [B, M, C, 1, 1]
        fea_v = (feas * attention_vectors).sum(dim=1)  # [B, C, H, W]
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, stride=1, dropout_rate=0.1):
        super(SKUnit, self).__init__()
        mid_features = out_features // 2
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),  # Dropout after first conv
            SKConv(mid_features, WH, M, G, r, stride=stride, dropout_rate=dropout_rate),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(p=dropout_rate)  # Dropout at the end of the block
        )
        self.shortcut = (
            nn.Identity() if in_features == out_features else
            nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features),
                nn.Dropout2d(p=dropout_rate)  # Dropout in shortcut if present
            )
        )

    def forward(self, x):
        return self.feas(x) + self.shortcut(x)


class SKNetSeg(nn.Module):
    def __init__(self, in_channels=1, dropout_rate=0.1):
        super(SKNetSeg, self).__init__()
        self.initial_dropout = nn.Dropout2d(p=dropout_rate)  # Initial dropout
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),  # After first conv
            
            SKUnit(64, 128, WH=128, M=2, G=8, r=2, stride=2, dropout_rate=dropout_rate),
            SKUnit(128, 256, WH=64, M=2, G=8, r=2, stride=2, dropout_rate=dropout_rate),
            SKUnit(256, 512, WH=32, M=2, G=8, r=2, stride=2, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)  # Final dropout in encoder
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.initial_dropout(x)  # Apply initial dropout
        x = self.encoder(x)
        x = self.decoder(x)
        return x