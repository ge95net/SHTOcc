import time

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import open_dict
from torch.cuda.amp import autocast

from .. import build_from_configs, evaluation, models


class LitModule(L.LightningModule):

    def __init__(self, *, model, optimizer, scheduler,voxel_backbone,ckpt_path,phase,criterion=None, evaluator=None, **kwargs):
        super().__init__()
        
        self.model = build_from_configs(models, model, **kwargs)
  
        
       
        #self.pretrained_voxel_model = LitModule.load_from_checkpoint(self.pretrained_weights)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.voxel_backbone = voxel_backbone
        self.phase = phase
        # self.defaults = defaults
        # self.datasets = self.defaults['datasets']
        self.criterion = build_from_configs(nn, criterion) if criterion else self.model.loss
        self.train_evaluator = build_from_configs(evaluation, evaluator)
        self.test_evaluator = build_from_configs(evaluation, evaluator)
        if 'class_names' in kwargs:
            self.class_names = kwargs['class_names']
        
        self.datasets = kwargs['data']['datasets']['type']


        
        print('self.ckpt_path=',ckpt_path)
        print('self.phase=',phase)
        for name, param in self.model.named_parameters():

            if 'clip' in name:
                param.requires_grad = False
      
        if self.phase == "stage_2":
            new_weights = {}
            pretrained_weights = torch.load(ckpt_path)['state_dict']
            for key, value in pretrained_weights.items():
                # 删除前缀 "models."
                new_key = key.replace('model.', '', 1)  # 只替换第一个出现的 "models."
                new_weights[new_key] = value
            self.model.load_state_dict(new_weights)
            for name, param in self.model.named_parameters():

                if 'segmentation_head' in name:
                    
                    param.requires_grad = True
                else:
                    param.requires_grad = False


   

        # channels=model['channels']
        
        # num_classes= model['num_classes']
        # #image_shape= model['image_shape']
        
        # downsample_z = model['downsample_z']
        # volume_scale = model['volume_scale']
        # voxel_size= model['voxel_size']#*volume_scale
        # scene_size = model['scene_size']
        # scene_size= [s // volume_scale for s in model['scene_size']]
        # #scene_size[-1] //= downsample_z

        # #downsample_z = 1
        # volume_scale = 1
        # volume_scale_data = 2
   
        # embed_dims = 128#model['embed_dims']
        # encoder = {'type': 'MMDetWrapper', 'config_path': 'maskdino/configs/maskdino_r50_8xb2-panoptic-export.py', 'custom_imports': 'maskdino', 'checkpoint_path': 'checkpoints/maskdino_r50_50e_300q_panoptic_pq53.0.pth'}
        
  
        # self.pretrained_voxel_model =  models.segmentors.Symphonies(embed_dims= embed_dims,channels=channels,scene_size=scene_size,
        #     num_classes=num_classes,downsample_z=downsample_z,volume_scale=volume_scale,volume_scale_data=volume_scale_data,
        #     num_layers=3,encoder=encoder,voxel_size=voxel_size,#depth=depth,
        #     view_scales=[4, 8, 16])
    
        # pretrained_weights = torch.load(pre_trained_weights)['state_dict']

        
        # new_weights = {}
        # for key, value in pretrained_weights.items():
        #         # 删除前缀 "models."
        #         new_key = key.replace('model.', '', 1)  # 只替换第一个出现的 "models."
        #         new_weights[new_key] = value

        # #self.pretrained_voxel_model =  self.pretrained_voxel_model.load_from_checkpoint(pre_trained_weights)
        # self.pretrained_voxel_model.load_state_dict(new_weights)

       
        # self.pretrained_voxel_model.cuda()
        # self.pretrained_voxel_model.eval()
        # for param in self.pretrained_voxel_model.parameters():
        #     param.requires_grad = False

    def forward(self, x,y=None):
        # with torch.no_grad():
        #     preds = self.pretrained_voxel_model(x)
        # inputs = {}
        # outs = preds['ssc_logits']
        # fov_mask = preds['fov_mask']
        # ref_vox = preds['ref_vox']
        
        # low_resolution_voxels = outs
     
        # inputs['low_resolution_voxels'] = low_resolution_voxels
        # inputs['fov_mask'] = fov_mask
        # inputs['ref_vox'] = ref_vox
  
        
        return self.model(x,y)

    def _step(self, batch, evaluator=None):
        x, y = batch

        torch.cuda.empty_cache()
        
        pred = self(x,y)

        with autocast(enabled=False):
            loss = self.criterion(pred, y)
        if evaluator:
            evaluator.update(pred, y)
        
        
        
        return loss

    def training_step(self, batch, batch_idx):


        loss = self._step(batch, self.train_evaluator)

        if isinstance(loss, dict):
            loss['loss_total'] = sum(loss.values())
            self.log_dict({f'train/{k}': v for k, v in loss.items()})
        else:
            self.log('train/loss', loss)
            
        # for name, param in self.model.named_parameters():

        #     if param.grad is not None and 'encoder' in name:
        #         print(f'Gradient for {name}: {param.grad}')
        
        def print_memory_usage(step):
            device = torch.device('cuda') 
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            print(f"{step} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            return allocated,reserved

        # optimizer = self.optimizers()
        # if isinstance(optimizer, list):  # 如果有多个优化器，取第一个
        #     optimizer = optimizer[0]
        
        # # 假设 Encoder 的参数在第一个参数组（根据你的实际配置调整索引）
        # encoder_lr = optimizer.param_groups[0]["lr"]
        
        # # 打印学习率（可选）
        # print(f"Encoder Learning Rate: {encoder_lr}")

        allocated,reserved = print_memory_usage("Whole Process")
        return sum(loss.values()) if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, 'test')

    def inference_step(self, batch):
        self._shared_eval(batch, 'val')

    def _shared_eval(self, batch, prefix):
        # print('-----------batch------------')
        # print(batch)  # 查看 batch 的内容
        # print(batch[0].keys())
        loss = self._step(batch, self.test_evaluator)
        # Lightning automatically accumulates the metric and averages it
        # if `self.log` is inside the `validation_step` and `test_step`

        if isinstance(loss, dict):
            loss['loss_total'] = sum(loss.values())
            self.log_dict({f'{prefix}/{k}': v for k, v in loss.items()}, sync_dist=True)
        else:
            self.log(f'{prefix}/loss', loss, sync_dist=True)

    def on_train_epoch_end(self):
        self._log_metrics(self.train_evaluator, 'train')

    def on_validation_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'val')

    def on_test_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'test')

    def on_inference_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'test')

    def _log_metrics(self, evaluator, prefix=None):
        metrics = evaluator.compute()
        iou_per_class = metrics.pop('iou_per_class')
        if prefix:
            metrics = {'/'.join((prefix, k)): v for k, v in metrics.items()}
        self.log_dict(metrics, sync_dist=True)

        if hasattr(self, 'class_names'):
            self.log_dict(
                {
                    f'{prefix}/iou_{c}': s.item()
                    for c, s in zip(self.class_names, iou_per_class)
                },
                sync_dist=True)
        evaluator.reset()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer
        scheduler_cfg = self.scheduler
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
            
        if self.phase=='stage_2':
            params = []
            params_name = []
            for k, v in self.named_parameters():
                if 'segmentation_head' in k:
                    params_name.append(k)
                    params.append(v)
        
        else:
            if paramwise_cfg:
                params = []
                pgs = [[] for _ in paramwise_cfg]

                for k, v in self.named_parameters():
                    in_param_group = False
                 
                    for i, pg_cfg in enumerate(paramwise_cfg):
                        if 'name' in pg_cfg and pg_cfg.name in k:
                            pgs[i].append(v)
                            in_param_group = True
                        # USER: Customize more cfgs if needed
                    if not in_param_group:
                        
                        params.append(v)
                
                        
            else:
                params = self.parameters()
       
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        if self.phase=='stage_1':
            if paramwise_cfg:
                for pg, pg_cfg in zip(pgs, paramwise_cfg):
                    cfg = {}
                    if 'lr_mult' in pg_cfg:
                        cfg['lr'] = optimizer_cfg.lr * pg_cfg.lr_mult
                    # USER: Customize more cfgs if needed
                    optimizer.add_param_group({'params': pg, **cfg})
        scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg.interval}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
