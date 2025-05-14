import os
import torch
import numpy as np
import random
os.environ['NCCL_P2P_DISABLE'] = '1'
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
torch.set_float32_matmul_precision('high')
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks

def set_seed(seed=0):
    """固定随机种子以确保实验的可重复性"""
    random.seed(seed)  # Python内置的random模块
    np.random.seed(seed)  # numpy库
    torch.manual_seed(seed)  # CPU上的PyTorch操作
    torch.cuda.manual_seed(seed)  # 当前GPU上的PyTorch操作
    torch.cuda.manual_seed_all(seed)  # 所有GPU上的PyTorch操作
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置为True可以增加运行效率
    os.environ['PYTHONHASHSEED'] = str(seed)  # 通过环境变量固定Python哈希算法的种子


@hydra.main(config_path='configs', config_name='config_stage2.yaml', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    set_seed(0)
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, **meta_info)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    
    if cfg.phase == 'stage_1':
        trainer = L.Trainer(strategy='ddp_find_unused_parameters_true', **cfg.trainer, **callbacks)
        
    else:
        trainer = L.Trainer(strategy="ddp_find_unused_parameters_true", **cfg.trainer, **callbacks)
    trainer.fit(model, *dls[:2])


if __name__ == '__main__':
    main()