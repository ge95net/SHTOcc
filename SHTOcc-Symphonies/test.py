import os
import os.path as osp
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import track
from tqdm import tqdm

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, build_from_configs, evaluation


def log_metrics(evaluator, prefix=None):
    metrics = evaluator.compute()
    iou_per_class = metrics.pop('iou_per_class')
    if prefix:
        metrics = {'/'.join((prefix, k)): v.item() for k, v in metrics.items()}
    print(f'metrics: {metrics}')

    ###

    class_names = ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
                   'table', 'tvs', 'furn', 'objs')

    iou_per_class_item = {
        f'{prefix}/iou_{c}': s.item()
        for c, s in zip(class_names, iou_per_class)
    }
    print(f'per class: {iou_per_class_item}')


    evaluator.reset()

def print_memory_usage(step):
        device = torch.device('cuda') 
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        #print(f"{step} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max memory: {peak_mem:.2f} MB")
        return allocated,reserved

@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[1]

    if cfg.get('ckpt_path'):
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, meta_info=meta_info)
    else:
        import warnings
        warnings.warn('\033[31;1m{}\033[0m'.format('No ckpt_path is provided'))
        model = LitModule(**cfg, meta_info=meta_info)

    test_evaluator = build_from_configs(evaluation, cfg.evaluator).cuda()
    model.cuda()
    model.eval()
    total_steps = len(data_loader)
    total_time = 0.0
    total_allocated = 0.0
    total_reserved= 0.0
    with torch.no_grad():
        for batch_inputs, targets in tqdm(data_loader):
            targets = {key: targets[key].cuda() for key in targets}
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            start_time = time.time()  # 开始计时
            outputs = model(batch_inputs)
            step_time = time.time() - start_time  # 计算每步所用的时间
            
            if test_evaluator:
                test_evaluator.update(outputs, targets)

            fps = 1 / step_time  # 计算FPS
            total_time += step_time
            print('step_time=',step_time,'fps=',fps)
            preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).astype(np.uint16)
            allocated,reserved = print_memory_usage('whole')
            total_allocated += allocated
            total_reserved += reserved
            # print(f"FPS: {fps:.2f}")

        log_metrics(test_evaluator, 'val')

        average_fps = total_steps / total_time  # 计算平均FPS
        average_allocated = total_allocated / total_steps
        average_reserved = total_reserved / total_steps
        print(f"Average FPS over {total_steps} steps: {average_fps:.2f}")
        print(f"Average allocated memory over {total_steps} steps: {average_allocated:.2f}")
        print(f"Average reserved memory over {total_steps} steps: {average_reserved:.2f}")
if __name__ == '__main__':
    main()


# python test.py --config-name config_cotr_swinl_tsdf_hvm.yaml +ckpt_path=/data/mqh/code/Indoor_occ/outputs/nyu_4cm/20241127-1219-cotr_swinl_4816_dim128_tsdf_hvm/version_0/e25_miou0.3162.ckpt +data_root=/data/mqh/code/Indoor_occ/data/nyuv2/NYU_dataset/depthbin +label_root=/data/mqh/code/Indoor_occ/data/nyuv2/NYU_dataset/preprocess/base +depth_root=/data/mqh/code/Indoor_occ/data/nyuv2/NYU_dataset/depthbin