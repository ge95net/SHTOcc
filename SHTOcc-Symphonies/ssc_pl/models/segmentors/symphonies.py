import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from ... import build_from_configs
from .. import encoders
#from ..head import SR_HEAD,SR_HEAD_BEV
from torch.distributions import normal
from ..decoders import SymphoniesDecoder
from ..losses import ce_ssc_loss,ce_ssc_loss_reweight, frustum_proportion_loss, geo_scal_loss, sem_scal_loss,BlvLoss,ce_ssc_loss_reweight,hvm_loss,vlgd_loss
# from depth_eval.depth_anything.dpt import DepthAnything
#from depth_eval.zoedepth.utils.config import get_config
#from depth_eval.zoedepth.models.builder import build_model
#from mono.utils.logger import setup_logger
import glob
#from mono.tools.test_scale_cano import parse_args as metric3D_config
#from mono.model.monodepth_model import get_configured_monodepth_model
#from mono.utils.running import load_ckpt
#from mono.utils.do_test import transform_test_data_scalecano,get_prediction
from ..Lseg import LSegNet
from ..layers import VLGD
import time
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
import torchvision.transforms as transforms
import clip
class Symphonies(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        image_shape=(370, 1220),
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        use_hvm=False,
        use_vlgd=False,
        LSeg_pth='ckpts/demo_e200.ckpt',
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions
        self.use_vlgd = use_vlgd
        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        self.decoder = SymphoniesDecoder(
            embed_dims,
            num_classes,
            num_layers=num_layers,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            use_hvm = use_hvm,
            downsample_z=downsample_z)
        self.use_hvm = use_hvm
        
 


    def forward(self, inputs,target=None):
        
 
        pred_insts = self.encoder(inputs['img'])
        

        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')
        

        
    
        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))

   
        decoder_outs = self.decoder(pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                            fov_mask,target)
       

   
        out_dict = {}
        if self.use_hvm and target!=None:
            outs, hvm_out_dict = decoder_outs[:2]
            out_dict.update({
                'ssc_logits': outs[-1],
                'refined_pred_tail_class': hvm_out_dict["refined_pred_tail_class"],
                'sampled_tail_target': hvm_out_dict["sampled_tail_target"],
                # 'refined_pred_head_class': hvm_out_dict["refined_pred_head_class"],
                # 'sampled_head_target': hvm_out_dict["sampled_head_target"],
                # 'refined_pred_empty_class': hvm_out_dict["refined_pred_empty_class"],
                # 'sampled_head_empty_target': hvm_out_dict["sampled_head_empty_target"],
            })
        else:

            outs, coordinates = decoder_outs[:2]
          
            out_dict.update({
                'ssc_logits': outs[-1],
                'aux_outputs': decoder_outs,
                'coordinates':coordinates,
            })

        # if self.use_vlgd:
        #     out_dict.update({
        #         'fused_feat_0': fused_feat_0,
        #         'pred_logits_0': pred_logits_0,
        #         'logits_per_image_0': logits_per_image_0,
        #         'sem_feat_0': feats[0],

        #         # 'fused_feat_1': fused_feat_1,
        #         # 'pred_logits_1': pred_logits_1,
        #         # 'logits_per_image_1': logits_per_image_1,
        #         # 'sem_feat_1': feats[1],

        #         # 'fused_feat_2': fused_feat_2,
        #         # 'pred_logits_2': pred_logits_2,
        #         # 'logits_per_image_2': logits_per_image_2,
        #         # 'sem_feat_2': feats[2]
        #     })

        return out_dict
            

    def loss(self, preds, target):
      
        loss_map = {
            'ce_ssc_re': ce_ssc_loss_reweight,
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss,
            'BlvLoss': BlvLoss,
            'hvm_loss': hvm_loss,
            'vlgd_loss': vlgd_loss
        }
        if 'BlvLoss' in self.criterions:
            cls_num_list = target['SEMANTIC_KITTI_CLASS_FREQ']
   
            cls_list = torch.cuda.FloatTensor(cls_num_list)
        
            frequency_list1 = torch.log(cls_list)

            frequency_list = torch.log(torch.sum(cls_num_list)) - frequency_list1
        
            sampler = normal.Normal(0, 4)
            pred1 = preds['ssc_logits'].float()
            
            viariation = sampler.sample(pred1.shape).clamp(-1, 1).to(pred1.device)
 
            
            frequency_list_expanded = frequency_list.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 变为 [1, 20, 1, 1, 1]
            frequency_list_expanded = frequency_list_expanded.expand(-1, -1, pred1.shape[2], pred1.shape[3], pred1.shape[4])  # 变为 [1, 20, 256, 256, 32]
            pred1 = pred1 + (viariation.abs()/ frequency_list_expanded.max() * frequency_list_expanded)
            preds['ssc_logits'] = pred1

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
  
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    if loss == 'ce_ssc_re':
                        losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                            'ssc_logits': pred
                        }, target) * scale
                    else:
                        losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                            'ssc_logits': pred
                        }, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = loss_map[loss](preds, target)


        # if self.use_hvm:
        #     assert 'refined_pred' in preds
        
        #     refined_pred = preds['refined_pred']
        #     gt_voxels= preds['sampled_target']
          
            
        #     print('refined_pred=',refined_pred)
        #     print('gt_voxels=',gt_voxels)
        #     ce_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        #     hvm_loss = ce_criterion(refined_pred.float(), gt_voxels.long())
        #     losses['tail_class_loss'] = hvm_loss 
    
        return losses

    # def loss(self, preds, target):
      
    #     loss_map = {
    #         'ce_ssc_re': ce_ssc_loss_reweight,
    #         'ce_ssc': ce_ssc_loss,
    #         'sem_scal': sem_scal_loss,
    #         'geo_scal': geo_scal_loss,
    #         'frustum': frustum_proportion_loss,
    #         'BlvLoss': BlvLoss
    #     }

    #     if 'BlvLoss' in self.criterions:
        
    #         from torch.distributions import normal
    #         cls_num_list = target['SEMANTIC_KITTI_CLASS_FREQ']
   
    #         cls_list = torch.cuda.FloatTensor(cls_num_list)
        
    #         frequency_list1 = torch.log(cls_list)

    #         frequency_list = torch.log(torch.sum(cls_num_list)) - frequency_list1
        
    #         sampler = normal.Normal(0, 4)
    #         pred1 = preds['ssc_logits'].float()
            
    #         viariation = sampler.sample(pred1.shape).clamp(-1, 1).to(pred1.device)
        
            
            
    #         frequency_list_expanded = frequency_list.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 变为 [1, 20, 1, 1, 1]
    #         frequency_list_expanded = frequency_list_expanded.expand(-1, -1, pred1.shape[2], pred1.shape[3], pred1.shape[4])  # 变为 [1, 20, 256, 256, 32]
    #         pred1 = pred1 + (viariation.abs()/ frequency_list_expanded.max() * frequency_list_expanded)
    #         preds['ssc_logits'] = pred1
    #     target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
     
    #     losses = {}
    #     if 'aux_outputs' in preds:
    #         for i, pred in enumerate(preds['aux_outputs']):
    #             scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
    #             for loss in self.criterions:
    #                 if loss == 'ce_ssc_re':
    #                     losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
    #                         'ssc_logits': pred
    #                     }, target) * scale*0.1
    #                 else:
    #                     losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
    #                         'ssc_logits': pred
    #                     }, target) * scale
    #     else:
    #         for loss in self.criterions:
    #             losses['loss_' + loss] = loss_map[loss](preds, target)
    #     return losses


