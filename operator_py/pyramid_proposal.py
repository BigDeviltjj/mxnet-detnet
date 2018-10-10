import mxnet as mx
import numpy as np
from distutils.util import strtobool
DEBUG = False
if DEBUG:
    from lib.bbox.bbox_transform import bbox_pred, clip_boxes
    from lib.rpn.generate_anchor import generate_anchors
    from lib.nms.nms import nms
else:
    from bbox.bbox_transform import bbox_pred, clip_boxes
    from rpn.generate_anchor import generate_anchors
    from nms.nms import nms


class PyramidProposalOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios, output_score, 
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size):
        super(PyramidProposalOperator,self).__init__()
        self._feat_stride = np.fromstring(feat_stride[1:-1],dtype = int, sep=',')
        self._scales = np.fromstring(scales[1:-1],dtype = float, sep = ',')
        self._ratios = np.fromstring(ratios[1:-1],dtype = float, sep = ',')
        self._num_anchors =  len(self._ratios)
        self._output_score = output_score
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self._threshold = threshold
        self._rpn_min_size = rpn_min_size
        
    @staticmethod
    def _clip_pad(tensor, pad_shape):
        H,W = tensor.shape[2:]
        h,w = pad_shape
        if h < H or w < W:
            tensor = tensor[:,:,:h,:w].copy()
        return tensor

    @staticmethod
    def _filter_boxes(boxes, min_size):
        ws = boxes[:,2] - boxes[:,0] + 1
        hs = boxes[:,3] - boxes[:,1] + 1
        keep = np.where((ws >= min_size)&(hs>=min_size))[0]
        return keep

    def forward(self,is_train, req, in_data, out_data, aux):
        batch_size = in_data[0].shape[0]

        if batch_size > 1:
            raise ValueError('sorry, only support single image')
        
        cls_prob_dict = {
            'stride64': in_data[4],
            'stride32': in_data[3],
            'stride16': in_data[2],
            'stride8': in_data[1],
            'stride4': in_data[0],
        }
        bbox_pred_dict = {
            'stride64': in_data[9],
            'stride32': in_data[8],
            'stride16': in_data[7],
            'stride8': in_data[6],
            'stride4': in_data[5],
        }
        #print(in_data[0])
        pre_nms_topN = self._rpn_pre_nms_top_n
        post_nms_topN = self._rpn_post_nms_top_n
        min_size = self._rpn_min_size

        proposal_list = []
        score_list = []
        for idx, s in enumerate(self._feat_stride):
            stride = int(s)
            sub_anchors = generate_anchors(base_size = stride, scales = [self._scales[idx]],ratios = self._ratios)
            scores = cls_prob_dict['stride'+str(s)].asnumpy()[:,self._num_anchors:,:,:]
            bbox_deltas = bbox_pred_dict['stride'+str(s)].asnumpy()
            im_info = in_data[-1].asnumpy()[0,:]

            #step 1
            height, width = int(im_info[0]/stride), int(im_info[1] / stride)
            shift_x = np.arange(width) * stride
            shift_y = np.arange(height) * stride
            shift_x, shift_y = np.meshgrid(shift_x,shift_y)
            shift = np.vstack([shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()]).transpose()
            #shift: K*4
            anchors = (sub_anchors[None,:,:] + shift[:,None,:]).reshape((-1,4))

            bbox_deltas = self._clip_pad(bbox_deltas, (height,width))
            bbox_deltas = bbox_deltas.transpose((0,2,3,1)).reshape((-1,4))

            scores = self._clip_pad(scores, (height, width))
            scores = scores.transpose((0,2,3,1)).reshape((-1,1))   #(1,W*W*A,1)

            proposals = bbox_pred(anchors, bbox_deltas)

            #step 2
            proposals = clip_boxes(proposals, im_info[:2])
            
            #step 3
            keep = self._filter_boxes(proposals, min_size * im_info[2])
            proposals = proposals[keep,:]
            scores = scores[keep]

            proposal_list.append(proposals)
            score_list.append(scores)
        proposals = np.vstack(proposal_list)
        scores = np.vstack(score_list)

        # step 4
        order = scores.ravel().argsort()[::-1]

        #step 5
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        

        proposals = proposals[order,:]
        scores=scores[order]

        #step 6
        det = np.hstack((proposals, scores)).astype(np.float32)
        keep = nms(det, self._threshold,in_data[0].context.device_id)
        #step 7
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        #step 8
        if len(keep) < post_nms_topN:
            pad = np.random.choice(keep, size=post_nms_topN - len(keep))
            keep  = np.hstack((keep,pad))
        proposals = proposals[keep,:]
        scores = scores[keep]

        batch_inds = np.zeros((proposals.shape[0],1), dtype = np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32,copy = False)))
        self.assign(out_data[0], req[0], blob)

        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy = False))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i],req[i],0)


@mx.operator.register('pyramid_proposal')
class PyramidProposalProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride = '(4,8,16,16,16)',scales = '(8,8,8,16,32)', ratios='(0.5,1,2)',output_score='False',
                 rpn_pre_nms_top_n='12000', rpn_post_nms_top_n='2000', threshold='0.3', rpn_min_size='16',output_pyramid_rois='False'):
        super(PyramidProposalProp, self).__init__()
        self._feat_stride = feat_stride
        self._scales = scales
        self._ratios = ratios
        self._output_score = strtobool(output_score)
        self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self._threshold = float(threshold)
        self._rpn_min_size = int(rpn_min_size)
        self.output_pyramid_rois = strtobool(output_pyramid_rois)

    def list_arguments(self):
        arg_list = []
        for i,s in enumerate(np.fromstring(self._feat_stride[1:-1],dtype=int,sep=',')):
            arg_list.append('rpn_cls_prob_p' + str(i+2))
        for i,s in enumerate(np.fromstring(self._feat_stride[1:-1],dtype=int,sep=',')):
            arg_list.append('rpn_bbox_pred_p' + str(i+2))
        arg_list.append('im_info')
        return arg_list
    
    def list_outputs(self):
        if self.output_pyramid_rois:
            return ['output', 'output_p3', 'output_p4', 'output_p5', 'output_idx']
        else:
            if self._output_score:
                return ['output','score']
            else:
                return ['output']

    def infer_shape(self, in_shape):
        output_shape = (self._rpn_post_nms_top_n,5)
        score_shape = (self._rpn_post_nms_top_n,1)
        #infershape:[argument_lists' shape] [outputs shape] []

        if self.output_pyramid_rois:
            return in_shape, [output_shape, output_shape, output_shape, output_shape, (self._rpn_post_nms_top_n,)]
        else:
            if self._output_score:
                return in_shape, [output_shape, score_shape]
            else:
                return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return PyramidProposalOperator(self._feat_stride, self._scales, self._ratios,self._output_score,
                                       self._rpn_pre_nms_top_n,self._rpn_post_nms_top_n,self._threshold,self._rpn_min_size)

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     return []


if __name__ == '__main__':
    if DEBUG:
        rpn_prob_p6 = mx.sym.Variable(name = 'rpn_cls_prob_stride64')
        rpn_prob_p5 = mx.sym.Variable(name = 'rpn_cls_prob_stride32')
        rpn_prob_p4 = mx.sym.Variable(name = 'rpn_cls_prob_stride16')
        rpn_prob_p3 = mx.sym.Variable(name = 'rpn_cls_prob_stride8')
        rpn_prob_p2 = mx.sym.Variable(name = 'rpn_cls_prob_stride4')

        rpn_bbox_pred_p6 = mx.sym.Variable(name = 'rpn_bbox_pred_stride64')
        rpn_bbox_pred_p5 = mx.sym.Variable(name = 'rpn_bbox_pred_stride32')
        rpn_bbox_pred_p4 = mx.sym.Variable(name = 'rpn_bbox_pred_stride16')
        rpn_bbox_pred_p3 = mx.sym.Variable(name = 'rpn_bbox_pred_stride8')
        rpn_bbox_pred_p2 = mx.sym.Variable(name = 'rpn_bbox_pred_stride4')

        im_info = mx.sym.Variable(name = 'im_info')
        rpn_cls_prob_dict = {
            'rpn_cls_prob_stride64': rpn_prob_p6,
            'rpn_cls_prob_stride32': rpn_prob_p5,
            'rpn_cls_prob_stride16': rpn_prob_p4,
            'rpn_cls_prob_stride8': rpn_prob_p3,
            'rpn_cls_prob_stride4': rpn_prob_p2,
        }
        rpn_bbox_pred_dict = {
            'rpn_bbox_pred_stride64': rpn_bbox_pred_p6,
            'rpn_bbox_pred_stride32': rpn_bbox_pred_p5,
            'rpn_bbox_pred_stride16': rpn_bbox_pred_p4,
            'rpn_bbox_pred_stride8': rpn_bbox_pred_p3,
            'rpn_bbox_pred_stride4': rpn_bbox_pred_p2,
        }
        arg_dict = {**rpn_cls_prob_dict,**rpn_bbox_pred_dict}
        aux_dict = {
            'op_type': 'pyramid_proposal', 'name': 'rois',
            'im_info': im_info, 
            'rpn_pre_nms_top_n': 12000, 'rpn_post_nms_top_n': 2000,
            'threshold': 0.7, 'rpn_min_size': 0
        }
        arg_shapes = {}
        img_shape = (768,1024)
        for k,v in arg_dict.items():
            if  'cls' in k:
                arg_shapes[k] = (1,6,int(img_shape[0] / int(k[(k.find('stride')+6):])),int(img_shape[1] / int(k[(k.find('stride')+6):])))
            else:
                arg_shapes[k] = (1,12,int(img_shape[0] / int(k[(k.find('stride')+6):])),int(img_shape[1] / int(k[(k.find('stride')+6):])))

        arg_shapes.update({'im_info':(1,3)})
        rois = mx.sym.Custom(**{**arg_dict,**aux_dict})
        print(rois.list_arguments())
        exe = rois.simple_bind(ctx=mx.gpu(),**arg_shapes)
        print(rois.infer_shape(**arg_shapes))
        np.random.seed(123)
        arg_val = {}
        arg_shapes = dict([(k,arg_shapes[k]) for k in sorted(arg_shapes.keys())] )
        for k,v in arg_shapes.items():
            arg_val[k] = mx.nd.array(np.random.randn(*v))
        arg_val['im_info'] = mx.nd.array([[768,1024,1.3]])
        print(arg_val)
        outputs = exe.forward(is_train = True,**arg_val)
        print(outputs[0])
        #print(outputs[0].shape)
