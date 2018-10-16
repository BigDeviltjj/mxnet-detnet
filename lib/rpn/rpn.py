import numpy as np
from utils.image import get_image
from rpn.generate_anchor import generate_anchors
from bbox.bbox_transform import bbox_overlaps, bbox_transform
import numpy.random as npr

def get_rpn_testbatch(roidb, cfg):
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']],dtype = np.float32) for i in range(len(roidb))]
    data = [{'data': im_array[i],
             'im_info': im_info[i]} for i in range(len(roidb))]
        
    label = {}
    return data, label, im_info

def get_rpn_batch(roidb,cfg):
    assert len(roidb) == 1, 'Single batch only'

    imgs, roidb = get_image(roidb,cfg)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype = np.float32)

    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0],5),dtype = np.float32)
        gt_boxes[:,:4] = roidb[0]['boxes'][gt_inds,:]
        gt_boxes[:,4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0,5),dtype = np.float32)

    data = {'data':im_array,
            'im_info':im_info}
    label = {'gt_boxes':gt_boxes}
    return data,label

def assign_pyramid_anchor(feat_shapes, gt_boxes, im_info, cfg, feat_strides=(4,8,16,16,16),
                          scales = (8,8,8,16,32),ratios = (0.5,1,2), allowed_border = 0, balance_scale_bg = False):
    def _unmap(data, count, inds, fill = 0):
        if len(data.shape) == 1:
            ret = np.empty((count,),dtype = np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:],dtype = np.float32)
            ret.fill(fill)
            ret[inds,:] = data
        return ret
    DEBUG = False
    im_info = im_info[0]
    scales = np.array(scales, dtype = np.float32)
    ratios = np.array(ratios, dtype = np.float32)
    fpn_args = []
    fpn_anchors_fid = np.zeros(0).astype(int)
    fpn_anchors = np.zeros([0,4])
    fpn_labels = np.zeros(0)
    fpn_inds_inside = []
    for feat_id in range(len(feat_strides)):
        base_anchors = generate_anchors(base_size = feat_strides[feat_id], ratios = ratios, scales = [scales[feat_id]])

        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shapes[feat_id][0][-2:]
        shift_x = np.arange(0, feat_width) * feat_strides[feat_id]
        shift_y = np.arange(0, feat_height) * feat_strides[feat_id]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
         
        A = num_anchors
        K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        labels = np.empty((len(inds_inside),),dtype = np.float32)
        labels.fill(-1)

        fpn_anchors_fid = np.hstack((fpn_anchors_fid,len(inds_inside)))
        fpn_anchors = np.vstack((fpn_anchors,anchors))
        fpn_labels = np.hstack((fpn_labels,labels))
        fpn_inds_inside.append(inds_inside)
        fpn_args.append([feat_height,feat_width,A,total_anchors])
    
    if gt_boxes.size > 0:
        overlaps = bbox_overlaps(fpn_anchors.astype(np.float),gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis = 1)
        max_overlaps = overlaps[np.arange(len(fpn_anchors)),argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis = 0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        fpn_labels[gt_argmax_overlaps] = 1
        fpn_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            fpn_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    else:
        fpn_labels[:] = 0

    num_fg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE ==-1 else int (cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(fpn_labels >= 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size = (len(fg_inds) - num_fg), replace = False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        fpn_labels[disable_inds] = -1
    
    num_bg = fpn_labels.shape[0] if cfg.TRAIN.RPN_BATCH_SIZE == -1 else cfg.TRAIN.RPN_BATCH_SIZE - np.sum(fpn_labels>=1)
    bg_inds = np.where(fpn_labels ==0)[0]
    fpn_anchors_fid = np.hstack((0,fpn_anchors_fid.cumsum()))

    if balance_scale_bg:
        num_bg_scale = num_bg / len(feat_strides)
        for feat_id in range(0,len(feat_strides)):
            bg_ind_scale = bg_inds[(bg_inds >= fpn_anchors_fid[feat_id]) & (bg_inds < fpn_anchors_fid[feat_id+1])]
            if len(bg_ind_scale) > num_bg_scale:
                disable_inds = npr.choice(bg_ind_scale, size=(len(bg_ind_scale) - num_bg_scale), replace=False)
                fpn_labels[disable_inds] = -1
    else:
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size = (len(bg_inds) - num_bg), replace = False)
            if DEBUG:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            fpn_labels[disable_inds] = -1

        
    fpn_bbox_targets = np.zeros((len(fpn_anchors),4),dtype = np.float32)
    if gt_boxes.size > 0:
        fpn_bbox_targets[fpn_labels>=1,:] = bbox_transform(fpn_anchors[fpn_labels>=1,:],gt_boxes[argmax_overlaps[fpn_labels >= 1], :4])
    
    fpn_bbox_weights = np.zeros((len(fpn_anchors),4),dtype = np.float32)
    fpn_bbox_weights[fpn_labels>=1,:] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    for feat_id in range(0,len(feat_strides)):
        feat_height, feat_width,A,total_anchors = fpn_args[feat_id]
        labels = _unmap(fpn_labels[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]],total_anchors,fpn_inds_inside[feat_id],fill = -1)
        bbox_targets = _unmap(fpn_bbox_targets[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)
        bbox_weights = _unmap(fpn_bbox_weights[fpn_anchors_fid[feat_id]:fpn_anchors_fid[feat_id+1]], total_anchors, fpn_inds_inside[feat_id], fill=0)

        labels = labels.reshape((1,feat_height, feat_width,A)).transpose(0,3,1,2)
        labels = labels.reshape((1,A*feat_height*feat_width))
        bbox_targets = bbox_targets.reshape((1,feat_height,feat_width,A*4)).transpose(0,3,1,2)
        bbox_targets = bbox_targets.reshape((1, A * 4, -1))
        bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
        bbox_weights = bbox_weights.reshape((1, A * 4, -1))

        label_list.append(labels)
        bbox_target_list.append(bbox_targets)
        bbox_weight_list.append(bbox_weights)

    label = {
        'label':np.concatenate(label_list,axis = 1),
        'bbox_target':np.concatenate(bbox_target_list, axis = 2),
        'bbox_weight':np.concatenate(bbox_weight_list,axis = 2)
    }

    return label#label['label'] = 1,(A*w1*h1+A*w2*h2 +...),label['bbox_target'] = (1,4A,(w1h1+w2h2+...))

