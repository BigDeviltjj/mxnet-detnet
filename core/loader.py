import mxnet as mx
from mxnet.executor_manager import _split_input_slice
import numpy as np

from rpn.rpn import get_rpn_batch, assign_pyramid_anchor

def par_assign_anchor_wrapper(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    data, rpn_label = get_rpn_batch(iroidb, cfg)
    data_shape = {k:v.shape for k,v in data.items()}
    del data_shape['im_info']
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis,:,:]
    feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
    label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'],data['im_info'],cfg,
                                  feat_strides, anchor_scales, anchor_ratios, allowed_border)
    return {'data':data,'label':label}
class PyramidAnchorIterator(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size = 1, shuffle = False, ctx = None, work_load_list = None,
                 feat_strides = (4,8,16,32,64), anchor_scales = (8,),anchor_ratios = (0.5,1,2),allowed_border = 0,
                 aspect_grouping = False):
        super(PyramidAnchorIterator,self).__init__()

        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size =batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.size = len(roidb)
        self.index = np.arange(self.size)

        if self.cfg.TRAIN.END2END:
            self.data_name = ['data','im_info','gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        self.label_name = ['label', 'bbox_target','bbox_weight']
        
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.reset()
        self.get_batch_parallel()
    
    @property
    def provide_data(self):
        return [(k,v.shape) for k,v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k,v.shape) for k,v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width']for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                ver_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds),np.random.permutation(ver_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra],(-1,self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm,:],(-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            self.cur += self.batch_size
            return mx.io.DataBatch(data = self.data, label = self.label,
                                   pad = self.getpad(), index = self.getindex(),
                                   provide_data = self.provide_data, provide_label = self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur // self.batch_size
    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self,max_data_shape = None, max_label_shape = None):
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2],max_shapes['data'][3],1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]   
        label = assign_pyramid_anchor(feat_shape, np.zeros((0,5)),im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, 
                                      self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k,tuple([input_batch_size] + list(v.shape[1:]))) for k,v in zip(self.label_name,label)]     

        return max_data_shape, label_shape
    def _combine_data(self,all_data):
        max_data_shape  = [len(all_data),0,0,0]
        max_gt_boxes_shape  = [len(all_data),0,0]
        for data in all_data:
          for i,dim in enumerate(data['data'].shape):
            max_data_shape[i] = max(max_data_shape[i],dim)
          for i,dim in enumerate(data['gt_boxes'].shape):
            max_gt_boxes_shape[i] = max(max_gt_boxes_shape[i],dim)
        img_data = mx.nd.zeros(tuple(max_data_shape))
        im_info = mx.nd.zeros(tuple([len(all_data),3]))
        gt_boxes = mx.nd.zeros(tuple(max_gt_boxes_shape))
        for i,data in enumerate(all_data):
           img_data[i,:data['data'].shape[1],:data['data'].shape[2],:data['data'].shape[3]] = mx.nd.array(data['data'][0]) 
           im_info[i] = mx.nd.array(data['im_info'][0]) 
           gt_boxes[i,:data['gt_boxes'].shape[1],:data['gt_boxes'].shape[2]] = mx.nd.array(data['gt_boxes'][0]) 
        self.data = [img_data,im_info,gt_boxes]
            
          
    def _combine_label(self,all_label):
        max_label_shape  = [len(all_label),0]
        max_bbox_target_shape  = [len(all_label),0,0]
        max_bbox_weight_shape  = [len(all_label),0,0]
        for label in all_label:
          for i,dim in enumerate(label['label'].shape):
            max_label_shape[i] = max(max_label_shape[i],dim)
          for i,dim in enumerate(label['bbox_target'].shape):
            max_bbox_target_shape[i] = max(max_bbox_target_shape[i],dim)
          for i,dim in enumerate(label['bbox_weight'].shape):
            max_bbox_weight_shape[i] = max(max_bbox_weight_shape[i],dim)
        label_data = mx.nd.full(tuple(max_label_shape),-1)
        bbox_target_data = mx.nd.zeros(tuple(max_bbox_target_shape))
        bbox_weight_data = mx.nd.zeros(tuple(max_bbox_weight_shape))
        for i,label in enumerate(all_label):
           label_data[i,:label['label'].shape[1]] = mx.nd.array(label['label'][0]) 
           bbox_target_data[i,:label['bbox_target'].shape[1],:label['bbox_target'].shape[2]] = mx.nd.array(label['bbox_target'][0]) 
           bbox_weight_data[i,:label['bbox_weight'].shape[1],:label['bbox_weight'].shape[2]] = mx.nd.array(label['bbox_weight'][0]) 
        self.label = [label_data,bbox_target_data, bbox_weight_data]
        
    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        slices = _split_input_slice(self.batch_size, work_load_list)

        max_data = {}
        max_label = {}
        data_lst = []
        rpn_label_lst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]

            data, rpn_label = get_rpn_batch(iroidb, self.cfg)
            data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis,:,:]
            data_shape = {k:v.shape for k,v in data.items()}
            if max_data == {} :
              max_data = data_shape
            else: 
              #max_data = {k:np.where(max_data[k]>v,max_data[k],v) for k,v in data_shape.items() }
              for k,v in data_shape.items():
                max_data[k] = np.where(np.array(max_data[k])>np.array(data_shape[k]),np.array(max_data[k]),np.array(data_shape[k]))
            data_lst.append(data)
            rpn_label_lst.append(rpn_label)
        for k,v in max_data.items():
          max_data[k][0] = self.batch_size
        self.data = [mx.nd.zeros(tuple(max_data['data'])),mx.nd.zeros(tuple(max_data['im_info'])),mx.nd.full(tuple(max_data['gt_boxes']),-1)]

        del max_data['im_info']
        del max_data['gt_boxes']

        max_data = {k:tuple(v) for k,v in max_data.items()}
        all_label = {}
        for idx, islice in enumerate(slices):
          feat_shape = [y[1] for y in [x.infer_shape(**max_data) for x in self.feat_sym]]
          d = data_lst[idx]
          
          self.data[0][idx,:d['data'].shape[1],:d['data'].shape[2],:d['data'].shape[3]] = d['data'][0]
          self.data[1][idx,:d['im_info'].shape[1]] = d['im_info'][0]
          self.data[2][idx,:d['gt_boxes'].shape[1],:d['gt_boxes'].shape[2]] = d['gt_boxes'][0]
           
          label = assign_pyramid_anchor(feat_shape, rpn_label_lst[idx]['gt_boxes'],data_lst[idx]['im_info'],self.cfg,
                                self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
          if all_label == {}:
            all_label = label
          else:
            for k,v in label.items():
              all_label[k] = np.vstack([all_label[k],v])
        self.label = [mx.nd.array(v) for k,v in all_label.items()]

       #     rst.append(par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides,
       #                                          self.anchor_scales, self.anchor_ratios, self.allowed_border))
        #all_data = [_['data'] for _ in rst]
        #all_label = [_['label'] for _ in rst]
        #self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        #self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        #self._combine_data(all_data)
        #self._combine_label(all_label)


        
