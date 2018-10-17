import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import pickle

from core.rcnn import sample_rois
import sys
import yaml

DEBUG = False
if DEBUG:
    sys.path.insert(0,'lib/')
    from easydict import EasyDict as edict

class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction):
        super(ProposalTargetOperator,self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._cfg = cfg
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0
    
    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()
        
        gt_keep = np.where(gt_boxes[:,-1]>0)[0]
        gt_boxes = gt_boxes[gt_keep]

        rois_per_image = int(self._batch_rois / self._batch_images)
        fg_rois_per_image =np.round(self._fg_fraction * rois_per_image).astype(int)
        zeros = np.zeros((gt_boxes.shape[0],1),dtype = gt_boxes.dtype)
        all_rois = np.vstack((all_rois,np.hstack((zeros,gt_boxes[:,:-1]))))
        #if DEBUG:
            #print(self._cfg.TRAIN.BATCH_ROIS)
        rois, labels ,bbox_targets, bbox_weights = \
            sample_rois(all_rois, fg_rois_per_image,rois_per_image,self._num_classes,self._cfg, gt_boxes = gt_boxes)

        if DEBUG:
            print("labels=", labels)
            print('num fg: {}'.format((labels > 0).sum()))
            print('num bg: {}'.format((labels == 0).sum()))
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print("self._count=", self._count)
            print('num fg avg: {}'.format(self._fg_num / self._count))
            print('num bg avg: {}'.format(self._bg_num / self._count))
            print('ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num)))

        for ind, val in enumerate([rois,labels,bbox_targets,bbox_weights]):
            self.assign(out_data[ind],req[ind],val)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i],req[i],0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes,batch_images,batch_rois,cfg,fg_fraction='0.25'):
        super(ProposalTargetProp, self).__init__()
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = pickle.loads(eval(cfg))
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output','label','bbox_target','bbox_weight']

    def infer_shape(self, in_shape):
        rpn_roi_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        rois = self._batch_rois

        output_rois_shape = (rois,5)
        label_shape = (rois,)
        bbox_target_shape = (rois,self._num_classes * 4)
        bbox_weight_shape = (rois,self._num_classes * 4)

        return [rpn_roi_shape, gt_boxes_shape],[output_rois_shape,label_shape,bbox_target_shape,bbox_weight_shape]

    def create_operator(self, ctx,shapes,dtypes):
        return ProposalTargetOperator(self._num_classes,self._batch_images, self._batch_rois, self._cfg,self._fg_fraction)

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     return []


if __name__ == '__main__':
    if DEBUG is  True:
        np.random.seed(223)
        gt_num = 10
        rpn_roi = np.random.uniform(0,600,size=(2000,4))
        rpn_label = np.zeros((2000,1))
        rpn_roi = np.hstack([rpn_label,rpn_roi])
        gt = np.zeros((gt_num,4))
        gt[:,0] = np.random.uniform(0,300,size=(gt_num))
        gt[:,1] = np.random.uniform(0,300,size=(gt_num))
        gt[:,2] = np.random.uniform(300,600,size=(gt_num))
        gt[:,3] = np.random.uniform(300,600,size=(gt_num))
        
        gt_label = np.random.randint(0,21,size=(gt_num,1))
        gt = np.hstack([gt,gt_label])
        rois = mx.sym.Variable('rois')
        gt_boxes = mx.sym.Variable('gt_boxes')
        with open('./cfgs/resnet_v1_101_coco_trainval_fpn_end2end_ohem.yaml') as f:
            cfg = edict(yaml.load(f))
        sym = mx.sym.Custom(rois = rois,gt_boxes = gt_boxes,op_type = 'proposal_target',
                            num_classes = cfg.dataset.NUM_CLASSES,  batch_images=cfg.TRAIN.BATCH_IMAGES,
                            batch_rois=cfg.TRAIN.BATCH_ROIS, cfg=pickle.dumps(cfg), fg_fraction=cfg.TRAIN.FG_FRACTION)

        exe = sym.simple_bind(ctx = mx.gpu(), rois = (2000,5),gt_boxes = (gt_num,5))

        outputs = exe.forward(is_train = True, rois = mx.nd.array(rpn_roi),gt_boxes = mx.nd.array(gt))
        print(outputs)
