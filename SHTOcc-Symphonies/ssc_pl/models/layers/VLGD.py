import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete



def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches

class ChannelAttention(nn.Module):
    """修正后的通道注意力模块（先GAP后MLP）"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP层实现
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        # 输入形状: (B, C, H, W)
        attn = self.gap(x)  # (B, C, 1, 1)
        attn = attn.view(attn.size(0), -1)  # (B, C)
        attn = self.mlp(attn)  # (B, C)
        attn = torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return attn

class VLGD(nn.Module):
    """修正后的Vision-Language Guidance Distillation模块"""
    def __init__(self, sem_channels, vision_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征融合层（修正后的实现）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sem_channels + vision_channels + num_classes, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, sem_channels, 1)
        )
        
        # 通道注意力模块
        self.vision_attn = ChannelAttention(sem_channels)
        self.sem_attn = ChannelAttention(sem_channels)
        
        # 新增的MLP投影层（对应论文公式3）
        self.mlp_vision = nn.Conv2d(sem_channels, sem_channels, 1)
        self.mlp_sem = nn.Conv2d(sem_channels, sem_channels, 1)
        
        # 语义预测头
        self.sem_head = nn.Sequential(
            ResidualBlock(sem_channels),
            ResidualBlock(sem_channels),
            nn.Conv2d(sem_channels, num_classes, 1)
        )
        
    def forward(self, sem_feat,logits, vision_feat):
        """
        输入:
            sem_feat (B, C, H, W): 图像编码器的语义特征
            vision_feat (B, C_v, H, W): VL模型的视觉特征
            text_feat (B, Q, C_t): VL模型的文本特征(Q为类别数)
        """
        B, C, H, W = sem_feat.shape
        
        # logits_up = F.interpolate(logits, 
        #                     size=(H*4, W*4), 
        #                     mode='bilinear', 
        #                     align_corners=False)
    
        # vision_up = F.interpolate(vision_feat, 
        #                         size=(H*4, W*4), 
        #                         mode='bilinear', 
        #                         align_corners=False)

        # 第二步：下采样到 (93, 305)
        logits = F.interpolate(logits, 
                                size=(H, W), 
                                mode='bilinear', 
                                align_corners=False)
        
        vision_feat = F.interpolate(vision_feat, 
                                size=(H, W), 
                                mode='bilinear', 
                                align_corners=False)


        logits = F.softmax(logits, dim=1)
        
        # Step 2: 特征拼接与融合（图4）
        fused = self.fusion_conv(torch.cat([
            sem_feat, 
            vision_feat, 
            logits  # 添加logits作为附加通道
        ], dim=1))  # (B, C, H, W)
        
        # Step 3: 通道注意力加权（公式2-3）
        vision_weight = self.vision_attn(fused)  # (B, C, 1, 1)
        sem_weight = self.sem_attn(sem_feat)     # (B, C, 1, 1)
        
        # 应用MLP投影（公式3）
        fused_proj = self.mlp_vision(fused)  # (B, C, H, W)
        sem_proj = self.mlp_sem(sem_feat)    # (B, C, H, W)
        
        # 加权融合
        fused_feat = (vision_weight * fused_proj) + (sem_weight * sem_proj)
        
        # Step 4: 计算特征蒸馏损失（公式4）
        #feat_loss = F.l1_loss(sem_feat, fused_feat)
        
        # Step 5: Logits蒸馏（公式5）
        pred_logits = self.sem_head(sem_feat)  # (B, Q, H, W)
        fused_feat_sem = self.sem_head(fused_feat)
        # # logits_loss = F.cross_entropy(
        # #     pred_logits, 
        # #     logits.argmax(dim=1)  # 使用argmax生成伪标签
        # # )
        
        
        labels = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
            'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
        
        out = F.interpolate(fused_feat_sem, 
                            size=(370, 1220), 
                            mode='bilinear', 
                            align_corners=False)
        predicts = [
            torch.max(out, 1)[1].cpu().numpy()
            for output in out
        ]
        
        pred = predicts[0]
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        fused = mask.convert("RGBA")
        fused.save('fused.png')
        
        
        out = F.interpolate(pred_logits, 
                            size=(370, 1220), 
                            mode='bilinear', 
                            align_corners=False)
        predicts = [
            torch.max(out, 1)[1].cpu().numpy()
            for output in out
        ]
        
        pred = predicts[0]
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        pred_logits = mask.convert("RGBA")
        pred_logits.save('pred_logits.png')
        
        
        return fused_feat,pred_logits,logits

class ResidualBlock(nn.Module):
    """保持不变的残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return F.relu(x + self.conv(x))