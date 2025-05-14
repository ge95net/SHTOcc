# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import time
import os

import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger

import mmcv
import numpy as np
from fvcore.nn import parameter_count_table
from projects.mmdet3d_plugin.utils import cm_to_ious, format_results, SSCMetrics

# utils for saving predictions 
from .utils import *

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3, pred_save=None, test_save=None):
    model.eval()
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    # evaluate ssc
    is_semkitti = hasattr(dataset, 'camera_used')
    ssc_metric = SSCMetrics().cuda()
    logger.info(parameter_count_table(model, max_depth=4))
    
    total_time = 0.0
    total_allocated = 0.0
    total_reserved= 0.0
    batch_size = 1
    total_steps = len(data_loader)
   
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            start_time = time.time()  # 开始计时
            result = model(return_loss=False, rescale=True, **data)
            step_time = time.time() - start_time  # 计算每步所用的时间

            fps = 1 / step_time  # 计算FPS
            total_time += step_time

            allocated,reserved = print_memory_usage('whole')
            total_allocated += allocated
            total_reserved += reserved
        
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
            
            # for one-gpu test, print results for each batch
            ious = cm_to_ious(evaluation_semantic)
            res_table, _ = format_results(ious, return_dic=True)
            print(res_table)
        
        img_metas = data['img_metas'].data[0][0]
        # save for test submission
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            target_voxels = result['target_voxels'].clone()
            ssc_metric.update(y_pred=output_voxels,  y_true=target_voxels)
            
            # compute metrics
            scores = ssc_metric.compute()
            if is_semkitti:
                print('\n Evaluating semanticKITTI occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            else:
                print('\n Evaluating nuScenes occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            
            # save for val predictions, mostly for visualization
            if pred_save is not None:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'], 
                        target_points=result['target_points'], 
                        save_path=pred_save, 
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))
        
        for _ in range(batch_size):
            prog_bar.update()
    
    res = {
        'ssc_scores': ssc_metric.compute(),
    }
    
    if type(evaluation_semantic) is np.ndarray:
        res['evaluation_semantic'] = evaluation_semantic
    
    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, pred_save=None, test_save=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    ssc_results = []
    ssc_results_refine = []
    ssc_metric = SSCMetrics().cuda()
    ssc_metric_refine = SSCMetrics().cuda()
    is_semkitti = hasattr(dataset, 'camera_used')
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    is_val_save_predictins = pred_save is not None
    if is_val_save_predictins:
        os.makedirs(pred_save, exist_ok=True)
    
    # evaluate lidarseg
    evaluation_semantic = 0
    

    total_time = 0.0
    total_allocated = 0.0
    total_reserved= 0.0
    batch_size = 1
    total_steps = len(data_loader)
   
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            start_time = time.time()  # 开始计时
            result = model(return_loss=False, rescale=True, **data)
            step_time = time.time() - start_time  # 计算每步所用的时间
            print('step time=',step_time)
            fps = 1 / step_time  # 计算FPS
            total_time += step_time

            allocated,reserved = print_memory_usage('whole')
            total_allocated += allocated
            total_reserved += reserved



            preds = torch.softmax(result['output_voxels'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1).astype(np.uint16)

            
            

            # 打印结果
            aLL_class = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            tail_class = [2,3,6,7,8]
            coordinates = outputs['coordinates']

            head_coordinates,head_coordnates_many,tail_coordinates,tail_coordinates_many = coordinates
            
            head_coordinates = head_coordnates_many.cpu().numpy()
            tail_coordinates = tail_coordinates_many.cpu().numpy()
            print('head_coordnates_many=',head_coordnates_many.shape)
            print('tail_coordinates_many=',tail_coordinates_many.shape)
            print('head_coordinates[:, 0]=',head_coordinates[:, 0].max())
            print('head_coordinates[:, 0]=',head_coordinates[:, 0].min())
            print('head_coordinates[:, 0]=',head_coordinates[:, 1].max())
            print('head_coordinates[:, 0]=',head_coordinates[:, 1].min())
            print('head_coordinates[:, 0]=',head_coordinates[:, 2].max())
            print('head_coordinates[:, 0]=',head_coordinates[:, 2].min())
    

            print('tail_coordinates[:, 0]=',tail_coordinates[:, 0].max())
            print('tail_coordinates[:, 0]=',tail_coordinates[:, 0].min())
            print('tail_coordinates[:, 0]=',tail_coordinates[:, 1].max())
            print('tail_coordinates[:, 0]=',tail_coordinates[:, 1].min())
            print('tail_coordinates[:, 0]=',tail_coordinates[:, 2].max())
            print('tail_coordinates[:, 0]=',tail_coordinates[:, 2].min())
            #mask_all = np.isin(preds[0, head_coordinates[:, 0], head_coordinates[:, 1], head_coordinates[:, 2]], aLL_class)
            #preds[0, head_coordinates[mask_all, 0], head_coordinates[mask_all, 1], head_coordinates[mask_all, 2]] = 20
            #preds[0, coords_last_4000[:, 0], coords_last_4000[:, 1], coords_last_4000[:, 2]] = 21
            mask = np.isin(preds[0, tail_coordinates[:, 0], tail_coordinates[:, 1], tail_coordinates[:, 2]], tail_class)
            preds[0, tail_coordinates[mask, 0], tail_coordinates[mask, 1], tail_coordinates[mask, 2]] = 21
            
            unique_values, counts = np.unique(preds, return_counts=True)

            # mask = (preds!= 0) & (preds != 21)
            # preds[mask] = 1
            for i in range(preds.shape[0]):
                output_dict = {'pred': preds[i]}
   

                output_dir = osp.join('outputs', 'SemanticKITTI','5')
                file_path = osp.join(output_dir, batch_inputs['frame_id'][i] + '.pkl')

                # keys = ('cam_pose', 'cam_K', 'voxel_origin', 'projected_pix_1', 'fov_mask_2')
           
                # for key in keys:
                #     if key == 'fov_mask_2':
                #         fov_mask = batch_inputs[key]
                #         print('fov_mask11=',fov_mask.shape)
                #         fov_mask = interpolate_flatten(
                #             fov_mask, [128,128,16], [128,128,8], mode='trilinear')
                #         output_dict[key] = fov_mask[i].detach().cpu().numpy()
                #         print('fov_mask22=',fov_mask.shape)
                #     else:
                        
                #         output_dict[key] = batch_inputs[key][i].detach().cpu().numpy()


                # keys_of_interest = []
                # for key in keys_of_interest:
                #     output_dict[key] = outputs[key].detach().cpu().numpy()

                os.makedirs(output_dir, exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(output_dict, f)
                    print('saved to', file_path)
        
        # # nusc lidar segmentation
        # if 'evaluation_semantic' in result:
        #     evaluation_semantic += result['evaluation_semantic']
        
        # img_metas = data['img_metas'].data[0][0]
        # # occupancy prediction
        # if is_test_submission:
        #     if is_semkitti:
        #         assert result['output_voxels'].shape[0] == 1
        #         save_output_semantic_kitti(result['output_voxels'][0], 
        #             test_save, img_metas['sequence'], img_metas['frame_id'])
        #     else:
        #         save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        # else:
        #     output_voxels = torch.argmax(result['output_voxels'], dim=1)
            
        #     if result['target_voxels'] is not None:
        #         target_voxels = result['target_voxels'].clone()
        #         ssc_results_i = ssc_metric.compute_single(
        #             y_pred=output_voxels, y_true=target_voxels)
        #         ssc_results.append(ssc_results_i)

        #         if result['output_voxel_refine'] is not None:
        #             output_voxels_refine = torch.argmax(result['output_voxel_refine'], dim=1)
        #             target_voxels = result['target_voxels'].clone()
        #             ssc_results_refine_i = ssc_metric_refine.compute_single(
        #                 y_pred=output_voxels_refine, y_true=target_voxels)
        #             ssc_results_refine.append(ssc_results_refine_i)
                    
            
        #     if is_val_save_predictins:
        #         if is_semkitti:
        #             # print(result['target_voxels'].shape)
        #             # print(result['output_voxels'].shape)
        #             save_output_semantic_kitti(result['target_voxels'][0], pred_save, 
        #                 img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
        #         else:
        #             save_occ_nusc(result['output_voxels'],
        #                           img_metas,
        #                           pred_save,
        #                           gt_occ=result['target_voxels'])
        #             # save_output_nuscenes(data['img_inputs'], output_voxels, 
        #             #     output_points=result['output_points'],
        #             #     target_points=result['target_points'], 
        #             #     save_path=pred_save,
        #             #     scene_token=img_metas['scene_token'], 
        #             #     sample_token=img_metas['sample_idx'],
        #             #     img_filenames=img_metas['img_filenames'],
        #             #     timestamp=img_metas['timestamp'],
        #             #     scene_name=img_metas.get('scene_name', None))
        
        # if rank == 0:
        #     for _ in range(batch_size * world_size):
        #         prog_bar.update()
    
    # wait until all predictions are generated
    dist.barrier()
    average_fps = total_steps / total_time  # 计算平均FPS
    average_allocated = total_allocated / total_steps
    average_reserved = total_reserved / total_steps
    print(f"Average FPS over {total_steps} steps: {average_fps:.2f}")
    print(f"Average allocated memory over {total_steps} steps: {average_allocated:.2f}")
    print(f"Average reserved memory over {total_steps} steps: {average_reserved:.2f}")
    if is_test_submission:
        return None
    
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    if result['output_voxel_refine'] is not None:
        # print(tmpdir)
        if tmpdir is None:
            tmpdir = '.refine'
        else:
            tmpdir = tmpdir.replace('.eval_hook', 'refine')
        # tmpdir = os.path.join(tmpdir, "refine")
        res['ssc_results_refine'] = collect_results_cpu(ssc_results_refine, len(dataset), tmpdir)
    
    if type(evaluation_semantic) is np.ndarray:
        # convert to tensor for reduce_sum
        evaluation_semantic = torch.from_numpy(evaluation_semantic).cuda()
        dist.all_reduce(evaluation_semantic, op=dist.ReduceOp.SUM)
        res['evaluation_semantic'] = evaluation_semantic.cpu().numpy()
    
    return res




def print_memory_usage(step):
    device = torch.device('cuda') 
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"{step} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max memory: {peak_mem:.2f} MB")
    return allocated,reserved