import torch
import torch.nn as nn

class TailVoxelSampler(nn.Module):
    def __init__(self, in_dim=128, num_classes=20, M=4096):
        super().__init__()
        self.M = M  # 需要采样的新体素数
        
        # 预测网络（全连接层）
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, voxel_feat, prev_indices,M, target_classes=[1,2,3,4,5,6,7,8]):#2,3,4,5,6,7,8,12,14,16,18,19
        """
        Args:
            voxel_feat: [1, N, 128] 体素特征
            prev_indices: [K] 已选体素索引
            target_classes: 目标类别列表
        Returns:
            new_indices: [M] 新采样索引
            combined_indices: [K+M] 合并后的唯一索引
        """
        self.M = M
        device = voxel_feat.device
        N = voxel_feat.shape[1]
        
        # 生成类别预测
        logits = self.cls_head(voxel_feat[0])  # [N, num_classes]
        
        # 构建剩余体素掩码
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        remaining_mask[prev_indices] = False
        
        # 筛选有效候选（剩余体素 & top2包含目标类）
        top2 = logits.topk(2).indices  # [N, 2]
        in_target = torch.isin(top2, torch.tensor(target_classes, device=device))
        valid_mask = remaining_mask & (in_target.any(dim=1))
        
        # 候选索引处理
        candidates = valid_mask.nonzero().squeeze(-1)
        if len(candidates) == 0:
            # 无候选时从剩余体素随机采样
            candidates = remaining_mask.nonzero().squeeze(-1)
        
        # 按目标类别最大logits排序
        target_logits = logits[candidates][:, target_classes].max(dim=1).values
        sorted_idx = candidates[target_logits.argsort(descending=True)]
        
        # 确定采样数量
        sample_num = min(self.M, len(sorted_idx))
        selected = sorted_idx[:sample_num]
        
        # 补充随机采样
        if sample_num < self.M:
            remaining = remaining_mask.nonzero().squeeze(-1)
            supplement = remaining[torch.randperm(len(remaining))[:self.M-sample_num]]
            selected = torch.cat([selected, supplement])
        
        # 合并索引并去重
        combined = torch.unique(torch.cat([prev_indices, selected]))
        return selected, combined