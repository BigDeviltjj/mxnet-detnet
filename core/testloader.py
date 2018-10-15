import numpy as np
import mxnet as mx
from rpn.rpn import get_rpn_testbatch
class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size = 1, shuffle = False,
                 has_rpn = False):
        super(TestLoader,self).__init__()

        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            raise ValueError
        
        self.label_name = None
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k,v.shape) for k,v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
        
    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.im_info, mx.io.DataBatch(data = self.data, label = self.label,
                        pad = self.getpad(),index = self.getindex(),
                        provide_data = self.provide_data,provide_label = self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur // self.batch_size
    
    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0
        
    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size,self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb, self.cfg)
        else:
            raise ValueError
        self.data = [[mx.nd.array(idata[name]) for name in self.data_name] for idata in data][0]
        self.im_info = im_info[0]

