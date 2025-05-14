import torch
import torch.nn.functional as F
from torch.distributions import normal



def ce_ssc_loss(pred, target):

    return F.cross_entropy(
        pred['ssc_logits'].float(),
        target['target'].long(),
        weight=target['class_weights'].float(),
        ignore_index=255,
        reduction='mean',
    )


def vlgd_loss(pred, target):
    fused_feat_0=pred['fused_feat_0']
    pred_logits_0=pred['pred_logits_0']
    logits_per_image_0=pred['logits_per_image_0']
    sem_feat_0=pred['sem_feat_0']
    
    # fused_feat_1=pred['fused_feat_1']
    # pred_logits_1=pred['pred_logits_1']
    # logits_per_image_1=pred['logits_per_image_1']
    # sem_feat_1=pred['sem_feat_1']

    # fused_feat_2=pred['fused_feat_2']
    # pred_logits_2=pred['pred_logits_2']
    # logits_per_image_2=pred['logits_per_image_2']
    # sem_feat_2=pred['sem_feat_2']

    feat_loss_0 = F.l1_loss(sem_feat_0, fused_feat_0)
    # feat_loss_1 = F.l1_loss(sem_feat_1, fused_feat_1)
    # feat_loss_2 = F.l1_loss(sem_feat_2, fused_feat_2)

    #feat_loss = (feat_loss_0 + feat_loss_1 +feat_loss_2 )/3
 

    logits_loss_0 = F.cross_entropy(
        pred_logits_0, 
        logits_per_image_0  # 使用argmax生成伪标签
    )

    # logits_loss_1 = F.cross_entropy(
    #     pred_logits_1, 
    #     logits_per_image_1.argmax(dim=1)  # 使用argmax生成伪标签
    # )

    # logits_loss_2 = F.cross_entropy(
    #     pred_logits_2, 
    #     logits_per_image_2.argmax(dim=1)  # 使用argmax生成伪标签
    # )
    # logits_loss = (logits_loss_0 + logits_loss_1 +logits_loss_2)/3

    return logits_loss_0#feat_loss_0 #+ logits_loss_0
    
    
def ce_ssc_loss_reweight(pred, target,sigma=0.98):

    #target['class_weights'][0] = target['class_weights'][0]/4
    # cls_num_list = target['SEMANTIC_KITTI_CLASS_FREQ']
   
    # cls_list = torch.cuda.FloatTensor(cls_num_list)
 
    # frequency_list1 = torch.log(cls_list)

    # frequency_list = torch.log(torch.sum(cls_num_list)) - frequency_list1
  
    # sampler = normal.Normal(0, sigma)
    # pred = pred['ssc_logits'].float()
    
    # viariation = sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)
  
    # softmax_tensor1 = F.softmax(pred[:,:,0,0,0], dim=1)
    
    # frequency_list_expanded = frequency_list.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 变为 [1, 20, 1, 1, 1]
    # frequency_list_expanded = frequency_list_expanded.expand(-1, -1, pred.shape[2], pred.shape[3], pred.shape[4])  # 变为 [1, 20, 256, 256, 32]
    # pred = pred + (viariation.abs()/ frequency_list_expanded.max() * frequency_list_expanded)
    return F.cross_entropy(
        pred['ssc_logits'].float(),
        target['target'].long(),
        weight=target['class_weights'].float(),
        ignore_index=255,
        reduction='mean',
        label_smoothing=0.07,
    )
    
def hvm_loss(pred, target,sigma=0.98):
    if 'refined_pred_tail_class' in pred:
        refined_pred = pred['refined_pred_tail_class']
        gt_voxels= pred['sampled_tail_target']
        tail_class_loss = F.cross_entropy(
                            refined_pred.float(),
                            gt_voxels.long(),
                            ignore_index=255,
                            reduction='mean',
                            
                        )
    else:
        tail_class_loss = 0
    if 'refined_pred_head_class' in pred:
        refined_pred = pred['refined_pred_head_class']
        gt_voxels= pred['sampled_head_target']
        head_class_loss = F.cross_entropy(
                            refined_pred.float(),
                            gt_voxels.long(),
                            ignore_index=255,
                            reduction='mean',
           
                        )
    else:
        head_class_loss = 0
        
    if 'refined_pred_empty_class' in pred:
        refined_pred = pred['refined_pred_empty_class']
        gt_voxels= pred['sampled_head_empty_target']
        empty_class_loss = F.cross_entropy(
                            refined_pred.float(),
                            gt_voxels.long(),
                            ignore_index=255,
                            reduction='mean',
                       
                        )
    else:
        empty_class_loss = 0
    return (tail_class_loss + head_class_loss + empty_class_loss)
# F.cross_entropy(
#         refined_pred.float(),
#         gt_voxels.long(),
#         ignore_index=255,
#         reduction='mean',
#     )
    
def BlvLoss(pred, target):
    

    return F.cross_entropy(
        pred['ssc_logits'].float(),
        target['target'].long(),
        weight=target['class_weights'].float(),
        ignore_index=255,
        reduction='mean',
    )
# class BlvLoss(nn.Module):
# #cls_nufrequency_list
#     def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
#         super(BlvLoss, self).__init__()
#         cls_list = torch.cuda.FloatTensor(cls_num_list)
#         frequency_list = torch.log(cls_list)
#         self.frequency_list = torch.log(sum(cls_num_list)) - frequency_list
#         self.reduction = 'mean'
#         self.sampler = normal.Normal(0, sigma)
#         self._loss_name = loss_name



#     def forward(self, pred, target, weight, ignore_index, avg_factor=None, reduction_override=None):

#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)

#         viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

#         pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)

#         loss = F.cross_entropy(pred, target, reduction='none',  ignore_index=ignore_index)

#         if weight is not None:
#             weight = weight.float()

#         loss = weight_reduce_loss(
#             loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

#         return loss

def sem_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target['target']
    mask = target != 255

    target = target[mask]

    loss, cnt = 0, 0
    num_classes = pred.shape[1]
    for i in range(0, num_classes):
        
        
        p = pred[:, i]
        p = p[mask]
        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0

        if torch.sum(completion_target) > 0:
            cnt += 1.0
            nominator = (p * completion_target).sum()
            
            if p.sum() > 0:
                precision = nominator / p.sum()
                loss += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if completion_target.sum() > 0:
                recall = nominator / completion_target.sum()
                loss += F.binary_cross_entropy(recall, torch.ones_like(recall))
            if (1 - completion_target).sum() > 0:
                specificity = (((1 - p) * (1 - completion_target)).sum() /
                               (1 - completion_target).sum())
                loss += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
    if cnt == 0:
        return 0
    return loss / cnt


def geo_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target['target']
    mask = target != 255

    empty_probs = pred[:, 0]
    # print('empty_probs.min: ', empty_probs.min())
    # print('empty_probs.max: ', empty_probs.max())
    nonempty_probs = 1 - empty_probs
    empty_probs = empty_probs[mask]
    nonempty_probs = nonempty_probs[mask]

    nonempty_target = target != 0
    nonempty_target = nonempty_target[mask].float()

    intersection = (nonempty_target * nonempty_probs).sum()
    loss = 0
    if nonempty_probs.sum() > 0:
        precision = intersection / nonempty_probs.sum()
        loss += F.binary_cross_entropy(precision, torch.ones_like(precision))
    if nonempty_target.sum() > 0:
        recall = intersection / nonempty_target.sum()
        loss += F.binary_cross_entropy(recall, torch.ones_like(recall))
    if (1 - nonempty_target).sum() > 0:
        specificity = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        loss += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
    return loss
    # return (F.binary_cross_entropy(precision, torch.ones_like(precision)) +
    #         F.binary_cross_entropy(recall, torch.ones_like(recall)) +
    #         F.binary_cross_entropy(specificity, torch.ones_like(specificity)))


def frustum_proportion_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)

    frustums_masks = target['frustums_masks']
    frustums_class_dists = target['frustums_class_dists']
    num_frustums = frustums_class_dists.shape[1]
    batch_cnt = frustums_class_dists.sum(0)  # n_fstm, n_cls

    frustum_loss = 0
    frustum_nonempty = 0
    for f in range(num_frustums):
        frustum_mask = frustums_masks[:, f].unsqueeze(1)
        prob = frustum_mask * pred  # bs, n_cls, H, W, D
        prob = prob.flatten(2).transpose(0, 1)
        prob = prob.flatten(1)  # n_cls, bs * H * W * D
        cum_prob = prob.sum(dim=1)  # n_cls

        total_cnt = batch_cnt[f].sum()
        total_prob = prob.sum()
        if total_prob > 0 and total_cnt > 0:
            fp_target = batch_cnt[f] / total_cnt
            cum_prob = cum_prob / total_prob

            nonzeros = fp_target != 0
            nonzero_p = cum_prob[nonzeros]
            frustum_loss += F.kl_div(torch.log(nonzero_p), fp_target[nonzeros], reduction='sum')
            frustum_nonempty += 1
    if frustum_nonempty == 0:
        return 0
    return frustum_loss / frustum_nonempty
    
