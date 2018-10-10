import mxnet as mx
import numpy as np
#from mxnet.contrib import autograd
import gc
DEBUG = True
class FPNROIPoolingOperator(mx.operator.CustomOp):
    def __init__(self, feat_strides, pooled_height, pooled_width, output_dim):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.feat_strides = feat_strides
        self.output_dim = output_dim
        self.in_grad_hist_list = []
        self.num_strides = len(self.feat_strides)
        self.roi_pool = [None for _ in range(self.num_strides)]
        self.feat_idx = [None for _ in range(self.num_strides)]
    
    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[-1].asnumpy()
        w = rois[:,3] - rois[:,1] + 1
        h = rois[:,4] - rois[:,2] + 1
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w*h)/224)),0,len(self.feat_strides) - 1)
        pyramid_idx = []

        rois_p = [None for _ in range(self.num_strides)]
        for i in range(self.num_strides):
            self.feat_idx[i] = np.where(feat_id == i)[0]  #第i个尺度的roi的下标
            if len(self.feat_idx[i]) == 0:
                rois_p[i] = np.zeros((1,5))
                pyramid_idx.append(-1)
            else:
                rois_p[i] = rois[self.feat_idx[i]]     #rois_p[i] 第i个尺度所有的roi
                pyramid_idx.append(self.feat_idx[i])  #pyramid[i]:第i个尺度的roi的下标
        rois_idx = np.argsort(np.hstack(pyramid_idx))[-rois.shape[0]:]
        if is_train:
            for i in range(self.num_strides):
                self.in_grad_hist_list.append(mx.nd.zeros_like(in_data[i]))

            mx.autograd.mark_variables([in_data[i] for i in range(self.num_strides)],self.in_grad_hist_list)
            with mx.autograd.record():
                for i in range(self.num_strides):
                    self.roi_pool[i] = mx.nd.ROIPooling(in_data[i],mx.nd.array(rois_p[i],in_data[i].context),(7,7),spatial_scale = 1.0 / self.feat_strides[i])
                roi_pool = mx.nd.concatenate(self.roi_pool, axis = 0)
        else:
            roi_pool = [None for _ in range(self.num_strides)]
            for i in range(self.num_strides):
                roi_pool[i] = mx.nd.ROIPooling(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0 / self.feat_strides[i])

            roi_pool = mx.nd.concatenate(roi_pool, axis=0)
        roi_pool = mx.nd.take(roi_pool, mx.nd.array(rois_idx, roi_pool.context))
        self.assign(out_data[0],req[0],roi_pool)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i],req[i],0)
        with mx.autograd.record():
            for i in range(self.num_strides):
                if len(self.feat_idx[i] > 0):
                    (mx.nd.take(out_grad[0],mx.nd.array(self.feat_idx[i],out_grad[0].context)) * self.roi_pool[i]).backward()
                    #autograd.compute_gradient([mx.nd.take(out_grad[0],mx.nd.array(self.feat_idx[i],out_grad[0].context)) * self.roi_pool[i]])
                    #autograd.compute_gradient([self.roi_pool[i]])


        for i in range(0,self.num_strides):
            self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])

        gc.collect()

@mx.operator.register('fpn_roi_pooling')
class FPNROIPoolingProp(mx.operator.CustomOpProp):
    def __init__(self, feat_strides='(4,8,16,16,16)',pooled_height = '7',pooled_width='7',output_dim='256'):
        super(FPNROIPoolingProp,self).__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.feat_strides = np.fromstring(feat_strides[1:-1],dtype = int, sep=',')
        self.output_dim = int(output_dim)
        self.num_strides = len(self.feat_strides)
    def list_arguments(self):
        args_list = []
        for i in range(self.num_strides):
            args_list.append('data_p{}'.format(2+i))

        args_list.append('rois')
        return args_list

    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        output_feat_shape = [in_shape[-1][0],in_shape[0][1],self.pooled_height,self.pooled_width]
        return in_shape,[output_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FPNROIPoolingOperator(self.feat_strides, self.pooled_height, self.pooled_width, self.output_dim)

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     return [out_grad[0]]

if __name__ == "__main__":
    if DEBUG:
        img_shape = np.array([768,1024])
        np.random.seed(225)
        fpn_p2 = mx.sym.Variable('data_p2')
        fpn_p3 = mx.sym.Variable('data_p3')
        fpn_p4 = mx.sym.Variable('data_p4')
        fpn_p5 = mx.sym.Variable('data_p5')
        rois = mx.sym.Variable('rois')
        data = {}
        data['data_p2'] = np.random.randn(256,*tuple(img_shape//4))[None,:,:,:]
        data['data_p3'] = np.random.randn(256,*tuple(img_shape//8))[None,:,:,:]
        data['data_p4'] = np.random.randn(256,*tuple(img_shape//16))[None,:,:,:]
        data['data_p5'] = np.random.randn(256,*tuple(img_shape//32))[None,:,:,:]
        data['rois'] = np.array([[0,0,56,56],[0,0,224,224],[0,0,224,224],
                                  [50,50,946,946]])
        data['rois'] = np.hstack([np.zeros((data['rois'].shape[0],1)),data['rois']])
        data_shape = {}
        for k,v in data.items():
            data_shape[k] = v.shape
        roi_pool = mx.sym.Custom(data_p2 = fpn_p2, data_p3 = fpn_p3, data_p4 = fpn_p4, data_p5 = fpn_p5,
                                 rois = rois, op_type = 'fpn_roi_pooling', name = 'fpn_roi_pooling')
        roi_pool = mx.sym.MakeLoss(roi_pool)
        exe = roi_pool.simple_bind(ctx = mx.cpu(),**data_shape)
        outputs = exe.forward(is_train = True,**data)
        exe.backward()
        print(outputs)
        g = exe.grad_arrays[0].asnumpy()
        print(np.sum(g==4.),np.sum(g==2.),np.sum(g==1.),np.sum(g==0.))
        #print(np.where(g==3.)[0])
        #print(np.where(g==2.)[0])
        #print(np.where(g==1.)[0])

        
