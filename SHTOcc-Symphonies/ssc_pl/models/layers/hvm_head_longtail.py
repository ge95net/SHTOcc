import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer

def voxel_sample(input, voxel_coords, **kwargs):
    """
    从输入张量中采样指定坐标的体素。
    Args:
        input (torch.Tensor): 输入张量，形状为 (B, C, H, W, Z)。
        voxel_coords (torch.Tensor): 体素坐标，形状为 (B, N, 3)。
    Returns:
        output (torch.Tensor): 采样后的张量，形状为 (B, C, N)。
    """
    add_dim = False
    if voxel_coords.dim() == 3:
        add_dim = True
        voxel_coords = voxel_coords.unsqueeze(2)
        voxel_coords = voxel_coords.unsqueeze(2)

    output = F.grid_sample(input, 2.0 * voxel_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(-1)
        output = output.squeeze(-1)
    return output

def sampling_target_voxels(
    target: torch.Tensor,
    target_classes: list = [1,2,3,4,5],
    N: int = 4096
) -> torch.Tensor:
    """
    使用 F.grid_sample 构建网格并采样目标类别体素，返回归一化坐标。
    
    Args:
        target (torch.Tensor): 目标体素标签，形状 (B, H, W, Z)
        pred (torch.Tensor): 预测体素概率（未使用，仅为兼容接口）
        target_classes (list): 目标类别列表
        N (int): 采样数量
    
    Returns:
        torch.Tensor: 归一化坐标，形状 (B, N, 3)
    """
    device = target.device
    B, H, W, Z = target.shape

    # 生成归一化网格 (B, H, W, Z, 3)
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, W, device=device),
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, Z, device=device),
        indexing='xy'
    )).permute(2, 1, 3, 0).unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, H, W, Z, 3)

    # 采样标签以对齐坐标空间
    sampled_labels = voxel_sample(target.float().unsqueeze(1), grid,mode="nearest", align_corners=False).squeeze(1)
    # sampled_labels = F.grid_sample(
    #     input=target.unsqueeze(1).float(),  # (B, 1, H, W, Z)
    #     grid=grid,
    #     mode='nearest',
    #     align_corners=True,
    #     padding_mode='border'
    # ).squeeze(1)  # (B, H, W, Z)

    # 生成目标掩码
    target_mask = torch.isin(sampled_labels, torch.tensor(target_classes, device=device))

    # 采样坐标
    batch_coords = []
    for b in range(B):
        # 提取当前批次的归一化坐标
        valid_coords = grid[b][target_mask[b]]  # (M, 3)

        # 处理无有效坐标的情况
        if valid_coords.size(0) == 0:
            coords = torch.zeros((N, 3), device=device)
        else:
            # 随机采样（允许重复）
            indices = torch.randint(0, valid_coords.size(0), (N,), device=device)
            coords = valid_coords[indices]

        batch_coords.append(coords.unsqueeze(0))

    return torch.cat(batch_coords, dim=0)

class HardVoxelMiningHead(nn.Module):
    def __init__(self, in_channel=532, embed_dims=64, num_classes=20):
        super().__init__()
        self.N = 4096  # 采样的体素数量
        self.refined_mlp = nn.Conv1d(in_channel, num_classes, 1)  # 细化预测的 MLP

    def forward(self, voxel_feat, coarse_prediction, target, tail_classes=[1,2,3,4,5,6,7,8],head_class=[1,9,10,11,13,15,17],empty_class=[0]):# [2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 19]
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入特征，未使用。
            voxel_feat (torch.Tensor): 体素特征，形状为 (B, C, H, W, Z)。
            coarse_prediction (torch.Tensor): 粗粒度预测，形状为 (B, num_classes, H, W, Z)。
            target (torch.Tensor): 目标标签，形状为 (B, H, W, Z)。
            target_classes (list): 需要采样的目标类别列表。
        Returns:
            output_dict (dict): 包含细化预测和采样坐标的字典。
        """
        output_dict = {}
        target = target['target']
        
        uniform_sampled_tail_coords = sampling_target_voxels(target, tail_classes, self.N)
        sampled_coarse_tail_pred = voxel_sample(coarse_prediction, uniform_sampled_tail_coords, align_corners=False)
        sampled_voxel_tail_feature = voxel_sample(voxel_feat, uniform_sampled_tail_coords, align_corners=False)
        sampled_tail_target = voxel_sample(target.float().unsqueeze(1), uniform_sampled_tail_coords,mode="nearest", align_corners=False).squeeze(1)
     
        sampled_voxel_feature = torch.cat([sampled_coarse_tail_pred, sampled_voxel_tail_feature], dim=1)
        output_dict["refined_pred_tail_class"] = self.refined_mlp(sampled_voxel_feature)
        output_dict["sampled_tail_target"] = sampled_tail_target


        # uniform_sampled_head_coords = sampling_target_voxels(target, head_class, self.N)
        # sampled_coarse_head_pred = voxel_sample(coarse_prediction, uniform_sampled_head_coords, align_corners=False)
        # sampled_voxel_head_feature = voxel_sample(voxel_feat, uniform_sampled_head_coords, align_corners=False)
        # sampled_tail_target = voxel_sample(target.float().unsqueeze(1), uniform_sampled_head_coords,mode="nearest", align_corners=False).squeeze(1)
     
        # sampled_voxel_feature = torch.cat([sampled_coarse_head_pred, sampled_voxel_head_feature], dim=1)
        # output_dict["refined_pred_head_class"] = self.refined_mlp(sampled_voxel_feature)
        # output_dict["sampled_head_target"] = sampled_tail_target

        # uniform_sampled_empty_coords = sampling_target_voxels(target, empty_class, self.N)
        # sampled_coarse_empty_pred = voxel_sample(coarse_prediction, uniform_sampled_empty_coords, align_corners=False)
        # sampled_voxel_empty_feature = voxel_sample(voxel_feat, uniform_sampled_empty_coords, align_corners=False)
        # sampled_tail_target = voxel_sample(target.float().unsqueeze(1), uniform_sampled_empty_coords,mode="nearest", align_corners=False).squeeze(1)
     
        # sampled_voxel_feature = torch.cat([sampled_coarse_empty_pred, sampled_voxel_empty_feature], dim=1)
        # output_dict["refined_pred_empty_class"] = self.refined_mlp(sampled_voxel_feature)
        # output_dict["sampled_head_empty_target"] = sampled_tail_target


        
        return output_dict