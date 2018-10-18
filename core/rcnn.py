import numpy as np
import numpy.random as npr
from lib.bbox.bbox_transform import bbox_overlaps,bbox_transform
from lib.bbox.bbox_regression import expand_bbox_regression_targets
import easydict
def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels = None, overlaps = None, bbox_targets = None, gt_boxes = None):
    if labels is None:
        overlaps = bbox_overlaps(rois[:,1:].astype(np.float),gt_boxes[:,:4].astype(np.float))
        gt_assignment = overlaps.argmax(axis = 1) #每个roi对应的最大的gt的id
        overlaps = overlaps.max(axis = 1)
        labels = gt_boxes[gt_assignment,4]    #求对应的label
    labels = labels.astype(np.int32)
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]#找出满足条件的正负样本并采样
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size = fg_rois_per_this_image, replace = False)

    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size = bg_rois_per_this_image, replace = False)
    keep_indexes = np.append(fg_indexes,bg_indexes)

    while keep_indexes.shape[0] < rois_per_image:     #一直补充到满足每个batch的长度
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)),size=gap, replace=False)
        keep_indexes = np.append(keep_indexes,gap_indexes)

    labels = labels[keep_indexes]
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes,:]
    else:
        targets = bbox_transform(rois[:,1:],gt_boxes[gt_assignment[keep_indexes],:4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))/ np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:,np.newaxis],targets))#[batch,5]
    
    bbox_targets, bbox_weights = expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)
    
    return rois, labels, bbox_targets,bbox_weights
