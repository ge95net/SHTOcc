from functools import reduce

import torch
import torch.nn.functional as F


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)


def cumprod(xs):
    return reduce(lambda x, y: x * y, xs)


def flatten_fov_from_voxels(x3d, fov_mask):
    assert x3d.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    return x3d.flatten(2)[..., fov_mask].transpose(1, 2)


def index_fov_back_to_voxels(x3d, fov, fov_mask, upsample_size=[256,256,32], head_indices=None,tail_indices=None):
    """
    增强版FOV到3D体素的映射函数，支持坐标索引转换
    
    Args:
        x3d (Tensor): 原始3D体素 [1, C, D, H, W]
        fov (Tensor): FOV特征 [1, C, N]
        fov_mask (Tensor): FOV掩码 [D*H*W] 或 [1, D*H*W]
        indices (Tensor, optional): 需要转换的FOV索引 [K]
    
    Returns:
        x3d_updated (Tensor): 更新后的3D体素 [1, C, D, H, W]
        coordinates (Tensor): 仅当indices存在时返回，形状 [K, 3] 的坐标矩阵
    """
    # --- 输入校验 ---
    assert x3d.shape[0] == fov.shape[0] == 1, "仅支持Batch=1的输入"
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1, "掩码的Batch维度应为1"
        fov_mask = fov_mask.squeeze(0)
    assert fov_mask.dim() == 1, "掩码必须为1D张量"
    
    # --- 核心FOV更新逻辑（保持原功能不变）---
    fov_concat = torch.zeros_like(x3d).flatten(2)  # [1, C, D*H*W]
    fov_concat[..., fov_mask] = fov.transpose(1, 2)  # 插入FOV特征
    x3d_updated = torch.where(
        fov_mask, 
        fov_concat, 
        x3d.flatten(2)
    ).reshape(*x3d.shape)
    head_coordinates,head_coordnates_many,tail_coordinates,tail_coordinates_many = None,None,None,None
    # --- 新增坐标转换功能 ---
    coords = None
    if head_indices is not None:
        # 获取3D体素维度
        _, _, D, H, W = x3d.shape
        # 128,128,8
        # 校验indices合法性
        N = fov_mask.sum().item()
        assert torch.all(head_indices < N), f"索引值超出范围(0-{N-1})"
        assert head_indices.dim() == 1, "索引必须为1D张量"
        
        # 获取FOV对应的线性索引
        flat_indices = torch.where(fov_mask)[0]  # [N]
        
        # 选择目标索引
        selected_flat = flat_indices[head_indices]  # [K]
        
        # 转换为3D坐标（向量化计算）
        d = selected_flat // (H * W)

        residual = selected_flat % (H * W)
        h = residual // W
        w = residual % W


        coords = torch.stack([d, h, w], dim=1)  # [K, 3]
        
        #return x3d_updated, coordinates

    if upsample_size is not None:
        
        
        # 坐标追踪
        if coords is not None:
            
            
            scale_factor = torch.tensor([
                upsample_size[0] / D,
                upsample_size[1] / H,
                upsample_size[2] / W
            ], device=coords.device)
 
            # 计算新坐标 (自动广播)
            coords1 = (coords.float() * scale_factor).round().long()
            
            # 边界检查
            head_coordinates = torch.clamp(coords1, min=torch.tensor(0, device=coords.device), max=torch.tensor([
                upsample_size[0]-1, 
                upsample_size[1]-1,
                upsample_size[2]-1
            ], device=coords.device))



    if upsample_size is not None and coords is not None:
        # 计算各轴步长（要求必须是整数）
        stride = torch.tensor([
            upsample_size[0] // D,
            upsample_size[1] // H,
            upsample_size[2] // W
        ], device=coords.device, dtype=torch.long)
      
        # 生成偏移量模板（三维组合）[sx*sy*sz, 3]
        offsets = torch.stack(torch.meshgrid(
            [torch.arange(s, device=coords.device) for s in stride],
            indexing='ij'
        ), dim=-1).view(-1, 3)

        # 分割原始坐标为两组
        group1 = coords[:int(len(coords)/2)]  # 第一组原始坐标
        group2 = coords[int(len(coords)/2):]  # 第二组原始坐标

        # 对每组分别上采样
        def process_group(group):
            # 计算基础坐标 [N,1,3] * [3] => [N,1,3]
            base = group[:, None, :] * stride
            # 生成所有可能偏移组合 [N, s^3, 3]
            new_coords = base + offsets[None, :, :]
            return new_coords.view(-1, 3)  # [N*s^3, 3]

        # 处理两个组并保持顺序
        head_coordnates_many = process_group(coords)
 
        # new_group2 = process_group(group2)
        # print('new_group2=',new_group2.shape)
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 0].max())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 0].min())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 1].max())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 1].min())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 2].max())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 2].min())
        # # 合并结果并保持原有分组顺序
        # coords = torch.cat([new_group1, new_group2], dim=0)

        # # 边界保护（可选）
        # # max_coords = torch.tensor(upsample_size, device=coords.device) - 1
        # # coords = torch.clamp(coords, min=torch.tensor(0, device=coords.device), max=max_coords)
        # print('coords=',coords.shape)
        # print('int(len(coords)/2)=',int(len(coords)/2))
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 0].max())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 0].min())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 1].max())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 1].min())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 2].max())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 2].min())

    # if upsample_size is not None:
    #     # 坐标追踪
    #     if coords is not None:
    #         # 计算各维度上采样倍数（步长）
    #         stride = torch.tensor([
    #             upsample_size[0] // D,
    #             upsample_size[1] // H,
    #             upsample_size[2] // W
    #         ], device=coords.device, dtype=torch.long)
            
    #         # 生成偏移量网格（产生新点的核心）
    #         offsets = torch.stack(torch.meshgrid(
    #             [torch.arange(s, device=coords.device) for s in stride],
    #             indexing='ij'
    #         ), dim=-1).reshape(-1, 3)  # [sx*sy*sz, 3]
            
    #         # 扩展原始坐标并生成新坐标
    #         base_coords = coords[:, None, :] * stride  # [N,1,3] * [3] => [N,1,3]
    #         new_coords = base_coords + offsets[None, :, :]  # [N,s^3,3]
    #         new_coords = new_coords.view(-1, 3)  # [N*s^3,3]
            
    #         # 边界保护（防止超出目标尺寸）
    #         max_coords = torch.tensor([
    #             upsample_size[0]-1, 
    #             upsample_size[1]-1,
    #             upsample_size[2]-1
    #         ], device=coords.device)
    #         new_coords = torch.clamp(new_coords, min=0, max=max_coords)
            
    #         # 去重（当多个原始点生成相同新点时）
    #         coords = torch.unique(new_coords, dim=0)



    # --- 新增坐标转换功能 ---
    coords = None
    if tail_indices is not None:
        # 获取3D体素维度
        _, _, D, H, W = x3d.shape
        # 128,128,8
        # 校验indices合法性
        N = fov_mask.sum().item()
        assert torch.all(tail_indices < N), f"索引值超出范围(0-{N-1})"
        assert tail_indices.dim() == 1, "索引必须为1D张量"
        
        # 获取FOV对应的线性索引
        flat_indices = torch.where(fov_mask)[0]  # [N]
        
        # 选择目标索引
        selected_flat = flat_indices[tail_indices]  # [K]
        
        # 转换为3D坐标（向量化计算）
        d = selected_flat // (H * W)

        residual = selected_flat % (H * W)
        h = residual // W
        w = residual % W


        coords = torch.stack([d, h, w], dim=1)  # [K, 3]
        
        #return x3d_updated, coordinates
        # print('coords=',coords.shape)

        # print('tail[:, 0] 000=',coords[:, 0].max())
        # print('tail[:, 0] 000=',coords[:, 0].min())
        # print('tail[:, 0] 000=',coords[:, 1].max())
        # print('tail[:, 0] 000=',coords[:, 1].min())
        # print('tail[:, 0] 000=',coords[:, 2].max())
        # print('tail[:, 0] 000=',coords[:, 2].min())
    if upsample_size is not None:
        
        
        # 坐标追踪
        if coords is not None:
            
            
            scale_factor = torch.tensor([
                upsample_size[0] / D,
                upsample_size[1] / H,
                upsample_size[2] / W
            ], device=coords.device)
           
            # 计算新坐标 (自动广播)
            coords1 = (coords.float() * scale_factor).round().long()
            
            # 边界检查
            tail_coordinates = torch.clamp(coords1, min=torch.tensor(0, device=coords.device), max=torch.tensor([
                upsample_size[0]-1, 
                upsample_size[1]-1,
                upsample_size[2]-1
            ], device=coords.device))
            # print('tail[:, 0] 111=',coords1[:, 0].max())
            # print('tail[:, 0] 111=',coords1[:, 0].min())
            # print('tail[:, 0] 111=',coords1[:, 1].max())
            # print('tail[:, 0] 111=',coords1[:, 1].min())
            # print('tail[:, 0] 111=',coords1[:, 2].max())
            # print('tail[:, 0] 111=',coords1[:, 2].min())


    if upsample_size is not None and coords is not None:
        # 计算各轴步长（要求必须是整数）
        stride = torch.tensor([
            upsample_size[0] // D,
            upsample_size[1] // H,
            upsample_size[2] // W
        ], device=coords.device, dtype=torch.long)

        # 生成偏移量模板（三维组合）[sx*sy*sz, 3]
        offsets = torch.stack(torch.meshgrid(
            [torch.arange(s, device=coords.device) for s in stride],
            indexing='ij'
        ), dim=-1).view(-1, 3)

        # 分割原始坐标为两组
        group1 = coords[:int(len(coords)/2)]  # 第一组原始坐标
        group2 = coords[int(len(coords)/2):]  # 第二组原始坐标

        # 对每组分别上采样
        def process_group(group):
            # 计算基础坐标 [N,1,3] * [3] => [N,1,3]
            base = group[:, None, :] * stride
            # 生成所有可能偏移组合 [N, s^3, 3]
            new_coords = base + offsets[None, :, :]
            return new_coords.view(-1, 3)  # [N*s^3, 3]

        # 处理两个组并保持顺序
        tail_coordinates_many = process_group(coords)
        # print('new_group1=',new_group1.shape)
        # print('tail[:, 0] 123=',new_group1[:, 0].max())
        # print('tail[:, 0] 123=',new_group1[:, 0].min())
        # print('tail[:, 0] 123=',new_group1[:, 1].max())
        # print('tail[:, 0] 123=',new_group1[:, 1].min())
        # print('tail[:, 0] 123=',new_group1[:, 2].max())
        # print('tail[:, 0] 123=',new_group1[:, 2].min())
        # new_group2 = process_group(group2)
        # print('new_group2=',new_group2.shape)
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 0].max())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 0].min())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 1].max())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 1].min())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 2].max())
        # print('coords_first_4000[:, 0] 321=',new_group2[:, 2].min())
        # # 合并结果并保持原有分组顺序
        # coords = torch.cat([new_group1, new_group2], dim=0)

        # # 边界保护（可选）
        # # max_coords = torch.tensor(upsample_size, device=coords.device) - 1
        # # coords = torch.clamp(coords, min=torch.tensor(0, device=coords.device), max=max_coords)
        # print('coords=',coords.shape)
        # print('int(len(coords)/2)=',int(len(coords)/2))
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 0].max())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 0].min())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 1].max())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 1].min())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 2].max())
        # print('coords_first_4000[:, 0] 222=',coords[:int(len(coords)/2), 2].min())

    # if upsample_size is not None:
    #     # 坐标追踪
    #     if coords is not None:
    #         # 计算各维度上采样倍数（步长）
    #         stride = torch.tensor([
    #             upsample_size[0] // D,
    #             upsample_size[1] // H,
    #             upsample_size[2] // W
    #         ], device=coords.device, dtype=torch.long)
            
    #         # 生成偏移量网格（产生新点的核心）
    #         offsets = torch.stack(torch.meshgrid(
    #             [torch.arange(s, device=coords.device) for s in stride],
    #             indexing='ij'
    #         ), dim=-1).reshape(-1, 3)  # [sx*sy*sz, 3]
            
    #         # 扩展原始坐标并生成新坐标
    #         base_coords = coords[:, None, :] * stride  # [N,1,3] * [3] => [N,1,3]
    #         new_coords = base_coords + offsets[None, :, :]  # [N,s^3,3]
    #         new_coords = new_coords.view(-1, 3)  # [N*s^3,3]
            
    #         # 边界保护（防止超出目标尺寸）
    #         max_coords = torch.tensor([
    #             upsample_size[0]-1, 
    #             upsample_size[1]-1,
    #             upsample_size[2]-1
    #         ], device=coords.device)
    #         new_coords = torch.clamp(new_coords, min=0, max=max_coords)
            
    #         # 去重（当多个原始点生成相同新点时）
    #         coords = torch.unique(new_coords, dim=0)

    return x3d_updated,[head_coordinates,head_coordnates_many,tail_coordinates,tail_coordinates_many]


def interpolate_flatten(x, src_shape, dst_shape, mode='nearest'):
    """Inputs & returns shape as [bs, n, (c)]
    """
    if len(x.shape) == 3:
        bs, n, c = x.shape
        x = x.transpose(1, 2)
    elif len(x.shape) == 2:
        bs, n, c = *x.shape, 1
    assert cumprod(src_shape) == n
    x = F.interpolate(
        x.reshape(bs, c, *src_shape).float(), dst_shape, mode=mode,
        align_corners=False).flatten(2).transpose(1, 2).to(x.dtype)
    if c == 1:
        x = x.squeeze(2)
    return x


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    assert L == cumprod(shape), 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def pix2cam(p_pix, depth, K):
    p_pix = torch.cat([p_pix * depth, depth], dim=1)  # bs, 3, h, w
    return K.inverse() @ p_pix.flatten(2)


def cam2vox(p_cam, E, vox_origin, vox_size, offset=0.5):
    p_wld = E.inverse() @ F.pad(p_cam, (0, 0, 0, 1), value=1)
    p_vox = (p_wld[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(1)) / vox_size - offset
    return p_vox


def pix2vox(p_pix, depth, K, E, vox_origin, vox_size, offset=0.5, downsample_z=1):
    p_cam = pix2cam(p_pix, depth, K)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size, offset)
    if downsample_z != 1:
        p_vox[..., -1] /= downsample_z
    return p_vox


def cam2pix(p_cam, K, image_shape):
    """
    Return:
        p_pix: (bs, H*W, 2)
    """
    p_pix = K @ p_cam / p_cam[:, 2]  # .clamp(min=1e-3)
    p_pix = p_pix[:, :2].transpose(1, 2) / (torch.tensor(image_shape[::-1]).to(p_pix) - 1)
    return p_pix


def vox2pix(p_vox, K, E, vox_origin, vox_size, image_shape, scene_shape):
    p_vox = p_vox.squeeze(2) * torch.tensor(scene_shape).to(p_vox) * vox_size + vox_origin
    p_cam = E @ F.pad(p_vox.transpose(1, 2), (0, 0, 0, 1), value=1)
    return cam2pix(p_cam[:, :-1], K, image_shape).clamp(0, 1)


def volume_rendering(
        volume,
        image_grid,
        K,
        E,
        vox_origin,
        vox_size,
        image_shape,
        depth_args=(2, 50, 1),
):
    depth = torch.arange(*depth_args).to(image_grid)  # (D,)
    p_pix = F.pad(image_grid, (0, 0, 0, 0, 0, 1), value=1)  # (B, 3, H, W)
    p_pix = p_pix.unsqueeze(-1) * depth.reshape(1, 1, 1, 1, -1)

    p_cam = K.inverse() @ p_pix.flatten(2)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size)
    p_vox = p_vox.reshape(1, *image_shape, depth.size(0), -1)  # (B, H, W, D, 3)
    p_vox = p_vox / (torch.tensor(volume.shape[-3:]) - 1).to(p_vox)

    return F.grid_sample(volume, torch.flip(p_vox, dims=[-1]) * 2 - 1, padding_mode='zeros'), depth


def render_depth(volume, image_grid, K, E, vox_origin, vox_size, image_shape, depth_args):
    sigmas, z = volume_rendering(volume, image_grid, K, E, vox_origin, vox_size, image_shape,
                                 depth_args)
    beta = z[1] - z[0]
    T = torch.exp(-torch.cumsum(F.pad(sigmas[..., :-1], (1, 0)) * beta, dim=-1))
    alpha = 1 - torch.exp(-sigmas * beta)
    depth_map = torch.sum(T * alpha * z, dim=-1).reshape(1, *image_shape)
    depth_map = depth_map  # + d[..., 0]
    return depth_map


def inverse_warp(img, image_grid, depth, pose, K, padding_mode='zeros'):
    """
    img: (B, 3, H, W)
    image_grid: (B, 2, H, W)
    depth: (B, H, W)
    pose: (B, 3, 4)
    """
    p_cam = pix2cam(image_grid, depth.unsqueeze(1), K)
    p_cam = (pose @ F.pad(p_cam, (0, 0, 0, 1), value=1))[:, :3]
    p_pix = cam2pix(p_cam, K, img.shape[2:])
    p_pix = p_pix.reshape(*depth.shape, 2) * 2 - 1
    projected_img = F.grid_sample(img, p_pix, padding_mode=padding_mode)
    valid_mask = p_pix.abs().max(dim=-1)[0] <= 1
    return projected_img, valid_mask
