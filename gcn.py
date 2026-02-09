import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Model.base import BaseModel
from Model.utils.helpers import initialize_weights
from itertools import chain
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights

'''
-> BackBone Resnet_GCN
'''

class Block_Resnet_GCN(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(Block_Resnet_GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), )
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu11 = nn.ReLU(inplace=False)
        self.conv12 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.relu12 = nn.ReLU(inplace=False)

        self.conv21 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu21 = nn.ReLU(inplace=False)
        self.conv22 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.relu22 = nn.ReLU(inplace=False)


    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)

        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)

        x = x1 + x2
        return x

class BottleneckGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_channels_gcn, stride=1):
        super(BottleneckGCN, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else: self.downsample = None
        
        self.gcn = Block_Resnet_GCN(kernel_size, in_channels, out_channels_gcn)
        self.conv1x1 = nn.Conv2d(out_channels_gcn, out_channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = self.gcn(x)
        x = self.conv1x1(x)
        x = self.bn1x1(x)

        x = x + identity
        return x

class ResnetGCN(nn.Module):
    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128), kernel_sizes=(5, 7)):
        super(ResnetGCN, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=False)

        if in_channels == 3: conv1 = resnet.conv1
        else: conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = nn.Sequential(
            BottleneckGCN(512, 1024, kernel_sizes[0], out_channels_gcn[0], stride=2),
            *[BottleneckGCN(1024, 1024, kernel_sizes[0], out_channels_gcn[0])]*5)
        self.layer4 = nn.Sequential(
            BottleneckGCN(1024, 2048, kernel_sizes[1], out_channels_gcn[1], stride=2),
            *[BottleneckGCN(1024, 1024, kernel_sizes[1], out_channels_gcn[1])]*5)
        initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = (x.size(2), x.size(3))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz

'''
-> BackBone Resnet
'''

class Resnet(nn.Module):
    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128),
                    pretrained=True, kernel_sizes=(5, 7)):
        super(Resnet, self).__init__()
        if backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None        
        resnet = getattr(torchvision.models, backbone)(weights=weights)

        if in_channels == 3: conv1 = resnet.conv1
        else: conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if not pretrained: initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = (x.size(2), x.size(3))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz

'''
-> Global Convolutionnal Network
'''

class GCN_Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout_rate=0.1):
        super(GCN_Block, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd'
        
        self.conv11 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.drop11 = nn.Dropout2d(p=dropout_rate)  # 添加Dropout
        self.conv12 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.drop12 = nn.Dropout2d(p=dropout_rate)  # 添加Dropout

        self.conv21 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.drop21 = nn.Dropout2d(p=dropout_rate)  # 添加Dropout
        self.conv22 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.drop22 = nn.Dropout2d(p=dropout_rate)  # 添加Dropout
        initialize_weights(self)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.drop11(x1)  # 应用Dropout
        x1 = self.conv12(x1)
        x1 = self.drop12(x1)  # 应用Dropout
        
        x2 = self.conv21(x)
        x2 = self.drop21(x2)  # 应用Dropout
        x2 = self.conv22(x2)
        x2 = self.drop22(x2)  # 应用Dropout

        x = x1 + x2
        return x

class BR_Block(nn.Module):
    def __init__(self, num_channels, dropout_rate=0.1):
        super(BR_Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.drop1 = nn.Dropout2d(p=dropout_rate)  # 添加Dropout
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.drop2 = nn.Dropout2d(p=dropout_rate)  # 添加Dropout
        self.relu2 = nn.ReLU(inplace=False)
        initialize_weights(self)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.drop1(x)  # 应用Dropout
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.drop2(x)  # 应用Dropout
        x = self.relu2(x)
        x = x + identity
        return x
    
def set_trainable(models, trainable):
    """
    设置模型组件的可训练性
    :param models: 模型组件列表（如 [backbone, head]）
    :param trainable: 布尔值，True 表示可训练，False 表示冻结
    """
    if not isinstance(models, list):
        models = [models]  # 确保输入是列表，方便统一处理
    for model in models:
        for param in model.parameters():
            param.requires_grad = trainable  # 设置参数是否需要计算梯度（即是否可训练）
class GCN(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, use_resnet_gcn=False, 
                backbone='resnet50', use_deconv=False, num_filters=11, freeze_bn=False,
                dropout_rate=0.1, **_):  # 新增dropout_rate参数
        super(GCN, self).__init__()
        self.use_deconv = use_deconv
        if use_resnet_gcn:
            self.backbone = ResnetGCN(in_channels, backbone=backbone)
        else:
            self.backbone = Resnet(in_channels, pretrained=pretrained, backbone=backbone)

        if (backbone == 'resnet34' or backbone == 'resnet18'): 
            resnet_channels = [64, 128, 256, 512]
        else: 
            resnet_channels = [256, 512, 1024, 2048]
        
        # 在GCN和BR模块中添加dropout_rate参数
        self.gcn1 = GCN_Block(num_filters, resnet_channels[0], num_classes, dropout_rate)
        self.br1 = BR_Block(num_classes, dropout_rate)
        self.gcn2 = GCN_Block(num_filters, resnet_channels[1], num_classes, dropout_rate)
        self.br2 = BR_Block(num_classes, dropout_rate)
        self.gcn3 = GCN_Block(num_filters, resnet_channels[2], num_classes, dropout_rate)
        self.br3 = BR_Block(num_classes, dropout_rate)
        self.gcn4 = GCN_Block(num_filters, resnet_channels[3], num_classes, dropout_rate)
        self.br4 = BR_Block(num_classes, dropout_rate)

        self.br5 = BR_Block(num_classes, dropout_rate)
        self.br6 = BR_Block(num_classes, dropout_rate)
        self.br7 = BR_Block(num_classes, dropout_rate)
        self.br8 = BR_Block(num_classes, dropout_rate)
        self.br9 = BR_Block(num_classes, dropout_rate)

        # 上采样部分也添加Dropout
        if self.use_deconv:
            self.decon1 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                 output_padding=1, stride=2, bias=False),
                nn.Dropout2d(p=dropout_rate)  # 添加Dropout
            )
            self.decon2 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                 output_padding=1, stride=2, bias=False),
                nn.Dropout2d(p=dropout_rate)  # 添加Dropout
            )
            self.decon3 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                 output_padding=1, stride=2, bias=False),
                nn.Dropout2d(p=dropout_rate)  # 添加Dropout
            )
            self.decon4 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                 output_padding=1, stride=2, bias=False),
                nn.Dropout2d(p=dropout_rate)  # 添加Dropout
            )
            self.decon5 = nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, padding=1,
                                 output_padding=1, stride=2, bias=False),
                nn.Dropout2d(p=dropout_rate)  # 添加Dropout
            )
        
        # 最终输出前添加Dropout
        self.final = nn.Sequential(
            nn.Dropout2d(p=dropout_rate/2),  # 输出层前使用较小的Dropout率
            nn.Conv2d(num_classes, num_classes, kernel_size=1)
        )
        
        if freeze_bn: 
            self.freeze_bn()
            set_trainable([self.backbone], False)

    def forward(self, x):
        x1, x2, x3, x4, conv1_sz = self.backbone(x)

        x1 = self.br1(self.gcn1(x1))
        x2 = self.br2(self.gcn2(x2))
        x3 = self.br3(self.gcn3(x3))
        x4 = self.br4(self.gcn4(x4))

        if self.use_deconv:
            # Padding处理尺寸对齐
            x4 = self.decon4(x4)
            if x4.size() != x3.size():
                x4 = self._pad(x4, x3)
            x3 = self.decon3(self.br5(x3 + x4))
            if x3.size() != x2.size():
                x3 = self._pad(x3, x2)
            x2 = self.decon2(self.br6(x2 + x3))
            x1 = self.decon1(self.br7(x1 + x2))

            x = self.br9(self.decon5(self.br8(x1)))
        else:
            x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
            x3 = F.interpolate(self.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
            x2 = F.interpolate(self.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = F.interpolate(self.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)

            x = self.br9(F.interpolate(self.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
        
        # 修复：将final_conv改为final
        return self.final(x)

    def _pad(self, x_topad, x):
        pad = (x.size(3) - x_topad.size(3), 0, x.size(2) - x_topad.size(2), 0)
        x_topad = F.pad(x_topad, pad, "constant", 0)
        return x_topad

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return [p for n, p in self.named_parameters() if 'backbone' not in n]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()