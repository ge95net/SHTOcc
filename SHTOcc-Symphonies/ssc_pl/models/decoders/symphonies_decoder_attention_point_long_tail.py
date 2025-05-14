import copy
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..layers import (ASPP, DeformableSqueezeAttention, DeformableTransformerLayer,TransformerLayer_attention,
                      LearnableSqueezePositionalEncoding, TransformerLayer, Upsample)
from ..projections import VoxelProposalLayer
from ..utils import (cumprod, flatten_fov_from_voxels, flatten_multi_scale_feats, generate_grid,
                     get_level_start_index, index_fov_back_to_voxels, interpolate_flatten,
                     nchw_to_nlc, nlc_to_nchw, pix2vox)


class SymphoniesLayer(nn.Module):

    def __init__(self, embed_dims,num_classes, num_heads=8, num_levels=3, num_points=4, query_update=True):
        super().__init__()
        
   
        self.num_levels = num_levels
   
        self.lift_feat_heads = nn.ModuleList()
        self.seg_pred_heads = nn.ModuleList()
        self.num_levels = num_levels
        
        for i in range(num_levels):
            
            self.lift_feat_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 8),
                nn.ReLU(inplace=True)
            ))
            
            
            self.seg_pred_heads.append(nn.Linear(embed_dims, num_classes))
            #self.occ_pred_heads.append(nn.Linear(embed_dims, 1))

       
        
        self.query_image_cross_defrom_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels, num_points)
        self.scene_query_cross_attn = TransformerLayer_attention(embed_dims, num_heads, mlp_ratio=0)
        
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)

        self.query_update = query_update
        if query_update:
            self.query_scene_cross_deform_attn =   DeformableTransformerLayer(
                embed_dims,
                num_heads,
                num_levels=1,
                num_points=num_points * 2,
                attn_layer=DeformableSqueezeAttention,
                mlp_ratio=0)
            self.query_self_attn = TransformerLayer_attention(embed_dims, num_heads)

    def forward(self,
                scene_embed,
                inst_queries,
                feats,
                scene_pos=None,
                inst_pos=None,
                ref_2d=None,
                ref_3d=None,
                ref_vox=None,
                fov_mask=None,
                
                i = None,
                query_coord=None,
                topk=[100, 400, 1600],
                scene_shape=None,
                bs=None,
                attention=None):

        # fov_mask = torch.ones_like(fov_mask, device=fov_mask.device)    # TODO
      
        #
    
        feats_flatten, feat_shapes = flatten_multi_scale_feats(feats)
       
        feats_level_index = get_level_start_index(feat_shapes)
        scene_pos_fov = flatten_fov_from_voxels(scene_pos,
                                                    fov_mask) if scene_pos is not None else None
        
        query_coord = query_coord.unsqueeze(0)
        
        scene_embed_fov = flatten_fov_from_voxels(scene_embed, fov_mask)
        
        if attention != None:
          
            max_values, _ = torch.max(attention, dim=2)
    
            top_values, top_indices = torch.topk(max_values, k=topk[i-1])
            top_indices = top_indices.squeeze(0)
 
     
            scene_pos_fov = scene_pos_fov[:,top_indices,:]
            scene_embed_fov_attention = scene_embed_fov[:,top_indices,:]
    
        else:
            scene_embed_fov_attention = scene_embed_fov
            top_indices = None
            scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
            scene_level_index = get_level_start_index(scene_shape)
        
            
        # new = torch.tensor([[1,1,topk[i]]],device=scene_embed_fov_attention.device)
        # new_level_index = get_level_start_index(new)
        
      
        
        
        
   
        inst_queries = self.query_image_cross_defrom_attn(               #instance-image-cross-attention
            inst_queries,
            feats_flatten,
            query_pos=inst_pos,
            ref_pts=ref_2d,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index)

        if i == 0:
            scene_embed_fov_attention,attention= self.scene_query_cross_attn(scene_embed_fov_attention, inst_queries, inst_queries,
                                                        scene_pos_fov, inst_pos)
       
        else:
            scene_embed_fov_attention,_= self.scene_query_cross_attn(scene_embed_fov_attention, inst_queries, inst_queries,
                                                        scene_pos_fov, inst_pos)
     
        # scene_embed_fov = self.scene_self_deform_attn(
        #     scene_embed_fov,
        #     scene_embed_flatten,
        #     query_pos=scene_pos_fov,
        #     ref_pts=torch.flip(current_ref_vox.float(), dims=[-1]),#torch.flip(ref_vox[:, fov_mask.squeeze()], dims=[-1]),  # TODO: assert bs == 1
        #     spatial_shapes=scene_shape,
        #     level_start_index=scene_level_index)
        
        if top_indices == None:
            ref_vox = ref_vox[:, fov_mask.squeeze()]
            
            scene_shape = torch.tensor([[1,1,scene_embed_fov_attention.shape[1]]],device=scene_embed_fov_attention.device)
            scene_level_index = get_level_start_index(scene_shape)
        else:
           
            
            ref_vox = ref_vox[:, fov_mask.squeeze()][:,top_indices,:,:]
        
            scene_shape = torch.tensor([[1,1,topk[i-1]]],device=scene_embed_fov_attention.device)
            scene_level_index = get_level_start_index(scene_shape)

        
        
        scene_embed_fov_attention = self.scene_self_deform_attn(
            scene_embed_fov_attention,
            scene_embed_fov_attention,
            query_pos=scene_pos_fov,
            ref_pts=torch.flip(ref_vox, dims=[-1]),  # TODO: assert bs == 1
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)

        
       
        #attention = None
        #scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        if not self.query_update:
            if top_indices != None:
                scene_embed_fov[:,top_indices,:] = scene_embed_fov_attention
            else:
                scene_embed_fov = scene_embed_fov_attention
            scene_embed = index_fov_back_to_voxels(scene_embed, scene_embed_fov, fov_mask)
       
            #scene_embed = sparse2dense(int_query_coord_2x_fov,scene_embed_fov,[scene_shape[0][0].item(),scene_shape[0][1].item(),scene_shape[0][2].item()],scene_embedded=scene_embed)
            return scene_embed, inst_queries,attention
       
     
        # inst_queries = self.query_scene_cross_deform_attn(
        #     inst_queries,
        #     scene_embed_flatten,
        #     query_pos=inst_pos,
        #     ref_pts=torch.flip(ref_3d, dims=[-1]),
        #     spatial_shapes=scene_shape,
        #     level_start_index=scene_level_index)
        
        

        inst_queries = self.query_scene_cross_deform_attn(
            inst_queries,
            scene_embed_fov_attention,
            query_pos=inst_pos,
            ref_pts=torch.flip(ref_3d, dims=[-1]),
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)
 
        inst_queries,_ = self.query_self_attn(inst_queries, query_pos=inst_pos)
        
        #scene_embed = sparse2dense(int_query_coord_2x_fov,scene_embed_fov,[scene_shape[0][0].item(),scene_shape[0][1].item(),scene_shape[0][2].item()],scene_embedded=scene_embed)
        if top_indices != None:
            scene_embed_fov[:,top_indices,:] = scene_embed_fov_attention
        else:
            scene_embed_fov = scene_embed_fov_attention
        scene_embed = index_fov_back_to_voxels(scene_embed, scene_embed_fov, fov_mask)
        return scene_embed, inst_queries,attention


class SymphoniesDecoder(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_layers,
                 num_levels,
                 scene_shape,
                 project_scale,
                 image_shape,
                 voxel_size=0.2,
                 use_tsdf=False,
                 use_hvm=False,
                 downsample_z=1):
        super().__init__()
        self.embed_dims = embed_dims
        scene_shape = [s // project_scale for s in scene_shape]
        if downsample_z != 1:
            self.ori_scene_shape = copy.copy(scene_shape)
            scene_shape[-1] //= downsample_z
        self.scene_shape = scene_shape
        self.num_queries = cumprod(scene_shape)
        self.image_shape = image_shape
        self.voxel_size = voxel_size * project_scale
     
        self.downsample_z = downsample_z
        self.upsample = False

        self.voxel_proposal = VoxelProposalLayer(embed_dims, scene_shape)
        self.layers = nn.ModuleList([
            SymphoniesLayer(
                embed_dims,
                num_classes=num_classes,
                num_levels=num_levels,
                query_update=True if i != num_layers - 1 else False) for i in range(num_layers)
        ])

        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        # self.scene_pos = LearnableSqueezePositionalEncoding((30,30,18),
        #                                                       embed_dims,
        #                                                       squeeze_dims=(1, 1, 1))
        self.scene_pos = LearnableSqueezePositionalEncoding((128, 128, 2),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))
        # self.scene_pos = LearnableSqueezePositionalEncoding((20, 20, 50),
        #                                                      embed_dims,
        #                                                      squeeze_dims=(5, 5, 1))
        # self.scene_pos = LearnableSqueezePositionalEncoding((50, 50, 25),
        #                                                     embed_dims,
        #                                                     squeeze_dims=(1, 1, 1))
        '''self.scene_pos = LearnableSqueezePositionalEncoding((128, 128, 2),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))'''

        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)
        voxel_grid = generate_grid(scene_shape, normalize=True)
        self.register_buffer('voxel_grid', voxel_grid)

        self.aspp = ASPP(embed_dims, (1, 3))
        assert project_scale in (1, 2)
        
    
        self.upsample= nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=(1, 1, downsample_z),
                    padding=1,
                    output_padding=(0, 0, downsample_z - 1),
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(),
            ) if downsample_z != 1 else nn.Identity(),
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            )
        self.segmentation_head1 = nn.Conv3d(embed_dims, num_classes, kernel_size=1)
        self.segmentation_head2 = nn.Conv3d(embed_dims, num_classes, kernel_size=1)
        self.segmentation_head3 = nn.Conv3d(embed_dims, num_classes, kernel_size=1)

        self.use_tsdf = use_tsdf
        if self.use_tsdf:
            pool_flag = False
            self.d1 = nn.Conv3d(1, embed_dims // 4, 3, stride=1, bias=True, padding=1)
            self.d2 = DDRBlock3D(embed_dims // 4, embed_dims // 2, embed_dims // 2, units=1, pool=pool_flag, residual=True, batch_norm=True, inst_norm=False)
            self.d_out = DDRBlock3D(embed_dims // 2, embed_dims, embed_dims, units=1, pool=pool_flag, residual=True, batch_norm=True, inst_norm=False)
            self.d_fuse = DDRBlock3D(embed_dims + embed_dims, embed_dims, embed_dims, units=1, pool=False, residual=True, batch_norm=True, inst_norm=False)


    @autocast(dtype=torch.float32)
    def forward(self, pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                fov_mask,target):
        inst_queries = pred_insts['queries']  # bs, n, c
        inst_pos = pred_insts.get('query_pos', None)
        bs = inst_queries.shape[0]
      
        if self.downsample_z != 1:
            projected_pix = interpolate_flatten(
                projected_pix, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            fov_mask = interpolate_flatten(
                fov_mask, self.ori_scene_shape, self.scene_shape, mode='trilinear')

        vol_pts = pix2vox(
            self.image_grid,
            depth.unsqueeze(1),
            K,
            E,
            voxel_origin,
            self.voxel_size,
            downsample_z=self.downsample_z).long()

        # print(f'vol_pts.shape: {vol_pts.shape}')


        ref_2d = pred_insts['pred_pts'].unsqueeze(2).expand(-1, -1, len(feats), -1)
        # print(f'ref_2d.shape: {ref_2d.shape}')
        ref_3d = self.generate_vol_ref_pts_from_masks(
            pred_insts['pred_boxes'], pred_masks,
            vol_pts).unsqueeze(2) if pred_masks else self.generate_vol_ref_pts_from_pts(
                pred_insts['pred_pts'], vol_pts).unsqueeze(2)
        # print(f'ref_3d.shape: {ref_3d.shape}')
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_pix = torch.flip(ref_pix, dims=[-1])
        ref_vox = nchw_to_nlc(self.voxel_grid.unsqueeze(0)).unsqueeze(2)

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        self.new_scene_embed = tensor = torch.randn(1, 20, 256, 256, 32, device=scene_embed.device)
        if self.use_tsdf and 'vox_tsdf' in pred_insts:
            vox_tsdf = pred_insts['vox_tsdf']
            vox_tsdf = self.d1(vox_tsdf)
            vox_tsdf = self.d2(vox_tsdf)
            vox_tsdf = self.d_out(vox_tsdf)
            # scene_embed += vox_tsdf  # add
            _x, _y, _z = vox_tsdf.shape[-3:]
            scene_embed_reshape = scene_embed.reshape(bs, _x, _y, _z, -1).permute(0, 4, 1, 2, 3)
            scene_embed_reshape = self.d_fuse(torch.cat([scene_embed_reshape, vox_tsdf], dim=1))  # cat
            scene_embed = scene_embed_reshape.flatten(2).permute(0, 2, 1)

  

        scene_pos = self.scene_pos().repeat(bs, 1, 1)
       
    
        scene_embed = self.voxel_proposal(scene_embed, feats, scene_pos, vol_pts, ref_pix)
        # print(f'scene_embed.shape after voxel_proposal: {scene_embed.shape}')

        scene_pos = nlc_to_nchw(scene_pos, self.scene_shape)


        outs = []
        
        attention = None
        #allocated,reserved = self.print_memory_usage("voxel_proposal")
        for i, layer in enumerate(self.layers):
     
           #[int(self.num_queries*0.1*0.25*0.25), int(self.num_queries*0.1*0.25), int(self.num_queries*0.1)] 
            scene_embed, inst_queries,attention = layer(scene_embed, inst_queries, feats, scene_pos, inst_pos,
                                              ref_2d, ref_3d, ref_vox, fov_mask,i,self.voxel_grid,topk=[1000,4000,16000],scene_shape=self.scene_shape,bs=bs,attention=attention)
            
      
            if i == 2:
                scene_embed = self.aspp(scene_embed)
            #if self.training or i == len(self.layers) - 1:
                scene_embed = self.upsample(scene_embed)
      

                scene_embed_1 = self.segmentation_head1(scene_embed[:,:,:,:,:10])
               
                scene_embed_2 = self.segmentation_head2(scene_embed[:,:,:,:,10:20])
                scene_embed_3 = self.segmentation_head3(scene_embed[:,:,:,:,20:32])


   
                self.new_scene_embed[:,:,:,:,:10] = scene_embed_1
                self.new_scene_embed[:,:,:,:,10:20] = scene_embed_2
                self.new_scene_embed[:,:,:,:,20:32] = scene_embed_3

         
                outs.append(self.new_scene_embed)
        
        #allocated,reserved = self.print_memory_usage("decoder")  
   
        return outs
    
    def print_memory_usage(self,step):
        device = torch.device('cuda') 
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"{step} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        return allocated,reserved

    def generate_vol_ref_pts_from_masks(self, pred_boxes, pred_masks, vol_pts):
        pred_boxes *= torch.tensor((self.image_shape + self.image_shape)[::-1]).to(pred_boxes)
        pred_pts = pred_boxes[..., :2].int()
        cx, cy, w, h = pred_boxes.split((1, 1, 1, 1), dim=-1)
        pred_boxes = torch.cat([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)],
                               dim=-1).int()
        pred_boxes[0::2] = pred_boxes[0::2].clamp(0, self.image_shape[1] - 1)
        pred_boxes[1::2] = pred_boxes[1::2].clamp(1, self.image_shape[1] - 1)

        pred_masks = F.interpolate(
            pred_masks.float(), self.image_shape, mode='bilinear').to(pred_masks.dtype)
        bs, n = pred_masks.shape[:2]

        for b, i in product(range(bs), range(n)):
            if pred_masks[b, i].sum().item() != 0:
                continue
            boxes = pred_boxes[b, i]
            pred_masks[b, i, boxes[1]:boxes[3], boxes[0]:boxes[2]] = True
            if pred_masks[b, i].sum().item() != 0:
                continue
            pred_masks[b, i, pred_pts[b, i, 1], pred_pts[b, i, 0]] = True
        pred_masks = pred_masks.flatten(2).unsqueeze(-1).to(vol_pts)  # bs, n, hw, 1
        vol_pts = vol_pts.unsqueeze(1) * pred_masks  # bs, n, hw, 3
        vol_pts = vol_pts.sum(dim=2) / pred_masks.sum(dim=2) / torch.tensor(
            self.scene_shape).to(vol_pts)
        return vol_pts.clamp(0, 1)

    def generate_vol_ref_pts_from_pts(self, pred_pts, vol_pts):
        pred_pts = pred_pts * torch.tensor(self.image_shape[::-1]).to(pred_pts)
        pred_pts = pred_pts.long()
        pred_pts = pred_pts[..., 1] * self.image_shape[1] + pred_pts[..., 0]

        assert pred_pts.size(0) == 1
        ref_pts = vol_pts[:, pred_pts.squeeze()]
      
        ref_pts = ref_pts / (torch.tensor(self.scene_shape) - 1).to(pred_pts)
   
        return ref_pts.clamp(0, 1)
