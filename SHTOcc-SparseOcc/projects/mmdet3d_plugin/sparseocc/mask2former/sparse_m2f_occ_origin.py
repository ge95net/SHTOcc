# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, Conv3d, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32

from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import (sample_valid_coords_with_frequencies,
                          get_uncertain_point_coords_3d_with_frequency,
                          preprocess_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies
from projects.mmdet3d_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss,hvm_loss
from projects.mmdet3d_plugin.utils.lovasz_softmax import lovasz_softmax

from einops import rearrange


# Sparse Mask2former Head for 3D Occupancy Segmentation
@HEADS.register_module()
class SparseMask2FormerOccHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 final_occ_size=[512, 512, 40],
                 num_queries=100,
                 num_transformer_feat_level=3,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 sample_weight_gamma=0.25,
                 empty_idx=0,
                 with_cp=True,
                 align_corners=True,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        self.with_cp = with_cp
        
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution, align the channel of input features
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv3d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        ''' Pixel Decoder Related, skipped '''
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.N = 4096  # 采样的体素数量
        self.refined_mlp = nn.Conv1d(feat_channels + self.num_classes, self.num_classes, 1)  # 细化预测的 MLP
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        # create class_weights for semantic_kitti
        self.class_weight = loss_cls.class_weight
        kitti_class_weights = 1 / np.log(semantic_kitti_class_frequencies)
        norm_kitti_class_weights = kitti_class_weights / kitti_class_weights[0]
        norm_kitti_class_weights = norm_kitti_class_weights.tolist()
        # append the class_weight for background
        norm_kitti_class_weights.append(self.class_weight[-1])
        self.class_weight = norm_kitti_class_weights
        
        loss_cls.class_weight = self.class_weight
        
        # computing sampling weight        
        sample_weights = 1 / semantic_kitti_class_frequencies
        sample_weights = sample_weights / sample_weights.min()
        self.baseline_sample_weights = sample_weights
        self.sample_weight_gamma = sample_weight_gamma
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.pooling_attn_mask = pooling_attn_mask
        
        # align_corners
        self.align_corners = align_corners

        # for sparse segmentation
        self.empty_idx = empty_idx
        #self.empty_token = nn.Embedding(1, out_channels)
        self.final_occ_size = final_occ_size
        self.occ_pred_conv = nn.Sequential(
            Conv3d(feat_channels, feat_channels//2, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, feat_channels//2),
            nn.ReLU(inplace=True),
            Conv3d(feat_channels//2, num_occupancy_classes, kernel_size=1, stride=1, padding=0)
        )

    def get_sampling_weights(self):
        if type(self.sample_weight_gamma) is list:
            # dynamic sampling weights
            min_gamma, max_gamma = self.sample_weight_gamma
            sample_weight_gamma = np.random.uniform(low=min_gamma, high=max_gamma)
        else:
            sample_weight_gamma = self.sample_weight_gamma
        
        self.sample_weights = self.baseline_sample_weights ** sample_weight_gamma
        
    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv3d):
                caffe2_xavier_init(m, bias=0)
        
        if hasattr(self, "pixel_decoder"):
            self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        
        # create sampling weights
        point_indices, point_coords = sample_valid_coords_with_frequencies(self.num_points, 
                gt_labels=gt_labels, gt_masks=gt_masks, sample_weights=self.sample_weights)
        
        point_coords = point_coords[..., [2, 1, 0]]
        mask_points_pred = point_sample_3d(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1), align_corners=self.align_corners).squeeze(1)
        
        # shape (num_gts, num_points)
        gt_points_masks = gt_masks.view(num_gts, -1)[:, point_indices]
        
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]
        
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
                gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, x, y, z).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, x, y, z).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)
        
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_weights = mask_weights[mask_weights > 0]
        
        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        
        with torch.no_grad():
            point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                mask_preds.unsqueeze(1), None, gt_labels_list, gt_masks_list, 
                self.sample_weights, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio)
            
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = torch.gather(mask_targets.view(mask_targets.shape[0], -1), 
                                        dim=1, index=point_indices)
        
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample_3d(
            mask_preds.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)
        
        # dice loss
        num_total_mask_weights = reduce_mean(mask_weights.sum())
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                        weight=mask_weights, avg_factor=num_total_mask_weights)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_weights = mask_weights.view(-1, 1).repeat(1, self.num_points)
        mask_point_weights = mask_point_weights.reshape(-1)
        
        num_total_mask_point_weights = reduce_mean(mask_point_weights.sum())
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight=mask_point_weights,
            avg_factor=num_total_mask_point_weights)

        return loss_cls, loss_mask, loss_dice

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size,indices=None):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries, x, y, z).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)
        
        ''' 对于一些样本数量较少的类别来说，经过 trilinear 插值 + 0.5 阈值，正样本直接消失 '''


        if indices is not None:
            h, w = attn_mask_target_size
            num_positions = h * w
            batch_size, num_queries = mask_pred.shape[0], mask_pred.shape[1]

            # 初始化全 True 掩码
            attn_mask = torch.ones((batch_size, num_queries, num_positions), 
                                dtype=torch.bool, device=mask_pred.device)

            # 校验索引范围
            assert indices.max() < num_positions, f"索引越界: {indices.max()} >= {num_positions}"

            # 生成三维索引（向量化操作）
            batch_idx = torch.arange(batch_size, device=indices.device)[:, None, None]  # (B, 1, 1)
            query_idx = torch.arange(num_queries, device=indices.device)[None, :, None]  # (1, Q, 1)
            attn_mask[batch_idx, query_idx, indices] = False  # 批量赋值
        else:

        
            if self.pooling_attn_mask:
                # however, using max-pooling can save more positive samples, which is quite important for rare classes
                attn_mask = F.adaptive_max_pool3d(mask_pred.float(), attn_mask_target_size)
            else:
                # by default, we use trilinear interp for downsampling
                attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='trilinear', align_corners=self.align_corners)
            
            # merge the dims of [x, y, z]
            attn_mask = attn_mask.flatten(2).detach() # detach the gradients back to mask_pred
            attn_mask = attn_mask.sigmoid() < 0.5
        
        # repeat for the num_head axis, (batch_size, num_queries, num_seq) -> (batch_size * num_head, num_queries, num_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)

        return cls_pred, mask_pred, attn_mask

    def preprocess_gt(self, gt_occ, img_metas):
        
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        targets = multi_apply(preprocess_occupancy_gt, gt_occ, num_class_list, img_metas)
        
        labels, masks = targets
        return labels, masks
    
    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            **kwargs,
        ):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        # reset the sampling weights
        self.get_sampling_weights()
        
        # forward
        all_cls_scores, all_mask_preds, coarse_occ,mask_features_detached = self.forward(voxel_feats, img_metas)
        
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_occ, img_metas)

        # loss
        loss_dict = {}
        losses_voxel = self.loss_voxel(coarse_occ, gt_occ)
        loss_dict.update(losses_voxel)

        hvm_out_dict = self.hvm_voxel(mask_features_detached,coarse_occ,gt_occ)
        tail_class_loss = hvm_loss(hvm_out_dict)

        loss_dict.update(tail_class_loss)
        
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
        loss_dict.update(losses)
        
        return loss_dict

    def forward(self, 
            voxel_feats,
            img_metas,
            **kwargs,
        ):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 5D-tensor (B, C, X, Y, Z).
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        """
        
        batch_size = len(img_metas)
        mask_features = voxel_feats[0]
        multi_scale_memorys = voxel_feats[:0:-1]
        
        decoder_inputs = []
        decoder_positional_encodings = []
        mask_features_detached = mask_features.detach().requires_grad_()
        if self.with_cp:
            coarse_occ = cp.checkpoint(self.occ_pred_conv, mask_features_detached)
        else:
            coarse_occ = self.occ_pred_conv(mask_features_detached)  # B, C, H, W, D
        

  
        for i in range(self.num_transformer_feat_level):
            ''' with flatten features '''
            # projection for input features
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            #print('decoder_input 111=',decoder_input.shape)
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            #print('decoder_input 222=',decoder_input.shape)
            ''' with level embeddings '''
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            #print('level_embed=',level_embed.shape)
            decoder_input = decoder_input + level_embed
            #print('decoder_input 333=',decoder_input.shape)
            ''' with positional encodings '''
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-3:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # for i,voxels in enumerate(decoder_inputs):
        #     print('i=',i,'decoder_inputs shape=',voxels.shape)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
      
        ''' directly deocde the learnable queries, as simple proposals '''
        cls_pred_list = []
        mask_pred_list = []
        cross_attention_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, 
                    mask_features, multi_scale_memorys[0].shape[-3:])
    
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
     
        for i in range(self.num_transformer_decoder_layers):

            
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            
            top_k = int(decoder_inputs[level_idx].shape[0] *0.6)
           
            if len(cross_attention_list)>=3:

                cross_attention = cross_attention_list[level_idx].permute(0,2,1)

                max_values, _ = torch.max(cross_attention, dim=2)
                
                top_values, top_indices = torch.topk(max_values, k=top_k)
                top_indices = top_indices.squeeze(0)
               
                selected, combined_indices = self.TailVoxelSampler(decoder_inputs[level_idx].permute(1,0,2),top_indices,M=top_k)
                #combined_indices = top_indices
                
                # decoder_positional_encodings_attention = decoder_positional_encodings[level_idx][combined_indices,:,:]
                # scene_embed_fov_attention = decoder_inputs[level_idx][combined_indices,:,:]
      
                # # 
                attn_masks_attention = torch.ones_like(attn_masks[0], dtype=torch.bool)

                # 将需要参与计算的位置设置为False（combined_indices对应位置不屏蔽）
                attn_masks_attention[:, :, combined_indices] = False

                
                
           
    
            else:
                
                combined_indices = None
                attn_masks_attention = attn_masks[0]

            attn_masks_attention[torch.where(
                attn_masks_attention.sum(-1) == attn_masks_attention.shape[-1])] = False
            attn_masks_attention = [attn_masks_attention, None]
            
            if i == 1 or i == 2 or i == 0:

                query_feat,cross_attention = layer(
                    query=query_feat,
                    key=decoder_inputs[level_idx],
                    value=decoder_inputs[level_idx],
                    query_pos=query_embed,
                    key_pos=decoder_positional_encodings[level_idx],
                    attn_masks=attn_masks_attention,
                    query_key_padding_mask=None,
                    key_padding_mask=None)
                cross_attention_list.append(cross_attention)
            else:
                query_feat,_ = layer(
                    query=query_feat,
                    key=decoder_inputs[level_idx],
                    value=decoder_inputs[level_idx],
                    query_pos=query_embed,
                    key_pos=decoder_positional_encodings[level_idx],
                    attn_masks=attn_masks_attention,
                    query_key_padding_mask=None,
                    key_padding_mask=None)
            
            


            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, 
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-3:])#,indices = combined_indices)
        

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        
        '''
        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        '''
        

        return cls_pred_list, mask_pred_list,coarse_occ,mask_features_detached

    def format_results(self, mask_cls_results, mask_pred_results):
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        output_voxels = torch.einsum("bqc, bqxyz->bcxyz", mask_cls, mask_pred)
        
        return output_voxels

    
    def simple_test(self, 
            voxel_feats,
            img_metas,
            **kwargs,
        ):
        all_cls_scores, all_mask_preds, coarse_occ,mask_features_detached = self.forward(voxel_feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=self.final_occ_size,
            mode='trilinear',
            align_corners=self.align_corners,
        )
        
        output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        res = {
            'output_voxels': [output_voxels],
            'output_voxel_refine': None,
            'output_points': None,
        }

        return res
    
    def loss_voxel(self, output_voxels, target_voxels, tag='coarse'):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        class_weights_tensor = torch.tensor(self.class_weight[:-1]).type_as(output_voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = CE_ssc_loss(output_voxels, target_voxels, class_weights_tensor, ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        return loss_dict
    

    

    def TailVoxelSampler(self, voxel_feat, prev_indices,M, target_classes=[2,5,7,8,10,12,14]):
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
       
        logits = self.cls_embed(voxel_feat[0])  # [N, num_classes]
     
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
    








    def voxel_sample(self,input, voxel_coords, **kwargs):
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
        self,
        target: torch.Tensor,
        target_classes: list = [2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 19],
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
        sampled_labels = self.voxel_sample(target.float().unsqueeze(1), grid,mode="nearest", align_corners=False).squeeze(1)
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



    def hvm_voxel(self, voxel_feat, coarse_prediction, target_voxels, tail_classes=[2,5,7,8,10,12,14]):# [2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 19]
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


        B, C, H, W, D = coarse_prediction.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(coarse_prediction).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        output_dict = {}
    
     
        uniform_sampled_tail_coords = self.sampling_target_voxels(target_voxels, tail_classes, self.N)
        sampled_coarse_tail_pred = self.voxel_sample(coarse_prediction, uniform_sampled_tail_coords, align_corners=False)
        sampled_voxel_tail_feature = self.voxel_sample(voxel_feat, uniform_sampled_tail_coords, align_corners=False)
        sampled_tail_target = self.voxel_sample(target_voxels.float().unsqueeze(1), uniform_sampled_tail_coords,mode="nearest", align_corners=False).squeeze(1)
    
        sampled_voxel_feature = torch.cat([sampled_coarse_tail_pred, sampled_voxel_tail_feature], dim=1)
        output_dict["refined_pred_tail_class"] = self.refined_mlp(sampled_voxel_feature)
        output_dict["sampled_tail_target"] = sampled_tail_target




            
        return output_dict