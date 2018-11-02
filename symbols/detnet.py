import pickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.pyramid_proposal import *
from operator_py.proposal_target import *
from operator_py.fpn_roi_pooling import *


class detnet(Symbol):
    def __init__(self):
        self.shared_param_list = ['rpn_conv', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_dict = {}
        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight')
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias')

    def get_detnet_backbone(self, data, is_train = True, with_dilated=True,  eps=1e-5):
        use_global_stats  = True#not is_train
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2), no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pad=(1, 1), kernel=(3, 3), stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=use_global_stats,
                                           fix_gamma=False, eps=eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(2, 2), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')

        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                    pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=use_global_stats,
                                           fix_gamma=False, eps=eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(2, 2), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=use_global_stats,
                                             fix_gamma=False, eps=eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')

        if with_dilated:
            res5_stride = (1, 1)
            res5_dilate = (2, 2)
        else:
            res5_stride = (2, 2)
            res5_dilate = (1, 1)

        # res5a-bottleneck
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=res5_stride, no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')

        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=256, pad=res5_dilate,
                                                kernel=(3, 3), stride=(1, 1), dilate=res5_dilate, no_bias=True)

        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5a_branch2c = bn5a_branch2c
        # res5a-shortcut
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b5_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=res5_stride, no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5a_branch1 = bn5a_branch1
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')

        # res5b-bottleneck
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')

        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=256, pad=res5_dilate,
                                                kernel=(3, 3), stride=(1, 1), dilate=res5_dilate, no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5b_branch2c = bn5b_branch2c
        # res5b-shortcut
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')

        # res5c-bottleneck
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')

        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=256, pad=res5_dilate,
                                                kernel=(3, 3), stride=(1, 1), dilate=res5_dilate, no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale5c_branch2c = bn5c_branch2c
        # res5c-shortcut
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        #stage 6
        if with_dilated:
            res6_stride = (1, 1)
            res6_dilate = (2, 2)
        else:
            res6_stride = (2, 2)
            res6_dilate = (1, 1)

        # res6a-bottleneck
        res6a_branch2a = mx.symbol.Convolution(name='res6a_branch2a', data=res5c_relu, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=res6_stride, no_bias=True)
        bn6a_branch2a = mx.symbol.BatchNorm(name='bn6a_branch2a', data=res6a_branch2a, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6a_branch2a = bn6a_branch2a
        res6a_branch2a_relu = mx.symbol.Activation(name='res6a_branch2a_relu', data=scale6a_branch2a, act_type='relu')

        res6a_branch2b = mx.symbol.Convolution(name='res6a_branch2b', data=res6a_branch2a_relu, num_filter=256, pad=res6_dilate,
                                                kernel=(3, 3), stride=(1, 1), dilate=res6_dilate, no_bias=True)

        bn6a_branch2b = mx.symbol.BatchNorm(name='bn6a_branch2b', data=res6a_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6a_branch2b = bn6a_branch2b
        res6a_branch2b_relu = mx.symbol.Activation(name='res6a_branch2b_relu', data=scale6a_branch2b, act_type='relu')
        res6a_branch2c = mx.symbol.Convolution(name='res6a_branch2c', data=res6a_branch2b_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn6a_branch2c = mx.symbol.BatchNorm(name='bn6a_branch2c', data=res6a_branch2c, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6a_branch2c = bn6a_branch2c
        # res6a-shortcut
        res6a_branch1 = mx.symbol.Convolution(name='res6a_branch1', data=res5c_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=res6_stride, no_bias=True)
        bn6a_branch1 = mx.symbol.BatchNorm(name='bn6a_branch1', data=res6a_branch1, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6a_branch1 = bn6a_branch1
        res6a = mx.symbol.broadcast_add(name='res6a', *[scale6a_branch1, scale6a_branch2c])
        res6a_relu = mx.symbol.Activation(name='res6a_relu', data=res6a, act_type='relu')

        # res6b-bottleneck
        res6b_branch2a = mx.symbol.Convolution(name='res6b_branch2a', data=res6a_relu, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn6b_branch2a = mx.symbol.BatchNorm(name='bn6b_branch2a', data=res6b_branch2a, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6b_branch2a = bn6b_branch2a
        res6b_branch2a_relu = mx.symbol.Activation(name='res6b_branch2a_relu', data=scale6b_branch2a, act_type='relu')

        res6b_branch2b = mx.symbol.Convolution(name='res6b_branch2b', data=res6b_branch2a_relu, num_filter=256, pad=res6_dilate,
                                                kernel=(3, 3), stride=(1, 1), dilate=res6_dilate, no_bias=True)
        bn6b_branch2b = mx.symbol.BatchNorm(name='bn6b_branch2b', data=res6b_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6b_branch2b = bn6b_branch2b
        res6b_branch2b_relu = mx.symbol.Activation(name='res6b_branch2b_relu', data=scale6b_branch2b, act_type='relu')
        res6b_branch2c = mx.symbol.Convolution(name='res6b_branch2c', data=res6b_branch2b_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn6b_branch2c = mx.symbol.BatchNorm(name='bn6b_branch2c', data=res6b_branch2c, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6b_branch2c = bn6b_branch2c
        # res6b-shortcut
        res6b = mx.symbol.broadcast_add(name='res6b', *[res6a_relu, scale6b_branch2c])
        res6b_relu = mx.symbol.Activation(name='res6b_relu', data=res6b, act_type='relu')

        # res6c-bottleneck
        res6c_branch2a = mx.symbol.Convolution(name='res6c_branch2a', data=res6b_relu, num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn6c_branch2a = mx.symbol.BatchNorm(name='bn6c_branch2a', data=res6c_branch2a, use_global_stats=use_global_stats,
                                            fix_gamma=False, eps=eps)
        scale6c_branch2a = bn6c_branch2a
        res6c_branch2a_relu = mx.symbol.Activation(name='res6c_branch2a_relu', data=scale6c_branch2a, act_type='relu')

        res6c_branch2b = mx.symbol.Convolution(name='res6c_branch2b', data=res6c_branch2a_relu, num_filter=256, pad=res6_dilate,
                                                kernel=(3, 3), stride=(1, 1), dilate=res6_dilate, no_bias=True)
        bn6c_branch2b = mx.symbol.BatchNorm(name='bn6c_branch2b', data=res6c_branch2b, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6c_branch2b = bn6c_branch2b
        res6c_branch2b_relu = mx.symbol.Activation(name='res6c_branch2b_relu', data=scale6c_branch2b, act_type='relu')
        res6c_branch2c = mx.symbol.Convolution(name='res6c_branch2c', data=res6c_branch2b_relu, num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn6c_branch2c = mx.symbol.BatchNorm(name='bn6c_branch2c', data=res6c_branch2c, use_global_stats=use_global_stats, fix_gamma=False, eps=eps)
        scale6c_branch2c = bn6c_branch2c
        # res6c-shortcut
        res6c = mx.symbol.broadcast_add(name='res6c', *[res6b_relu, scale6c_branch2c])
        res6c_relu = mx.symbol.Activation(name='res6c_relu', data=res6c, act_type='relu')


        return res2c_relu, res3b3_relu, res4b5_relu, res5c_relu, res6c_relu

    def get_fpn_feature(self, c2,c3,c4,c5,c6,feature_dim = 256):
        fpn_p6_1x1 = mx.sym.Convolution(data=c6, kernel=(1,1),pad=(0,0),stride = (1,1),num_filter = feature_dim, name = 'fpn_p6_1x1')
        fpn_p5_1x1 = mx.sym.Convolution(data=c5, kernel=(1,1),pad=(0,0),stride = (1,1),num_filter = feature_dim, name = 'fpn_p5_1x1')
        fpn_p4_1x1 = mx.sym.Convolution(data=c4, kernel=(1,1),pad=(0,0),stride = (1,1),num_filter = feature_dim, name = 'fpn_p4_1x1')
        fpn_p3_1x1 = mx.sym.Convolution(data=c3, kernel=(1,1),pad=(0,0),stride = (1,1),num_filter = feature_dim, name = 'fpn_p3_1x1')
        fpn_p2_1x1 = mx.sym.Convolution(data=c2, kernel=(1,1),pad=(0,0),stride = (1,1),num_filter = feature_dim, name = 'fpn_p2_1x1')
        
        fpn_p5_plus = mx.sym.ElementWiseSum(*[fpn_p5_1x1, fpn_p6_1x1],name = 'fpn_p5_sum')
        fpn_p4_plus = mx.sym.ElementWiseSum(*[fpn_p5_plus, fpn_p4_1x1],name = 'fpn_p4_sum')
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale = 2, sample_type = 'nearest', name = 'fpn_p4_upsample')
        fpn_p3_plus = mx.sym.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1],name = 'fpn_p3_sum')
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale = 2, sample_type = 'nearest', name = 'fpn_p3_upsample')
        fpn_p2_plus = mx.sym.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1],name = 'fpn_p2_sum')

        fpn_p6 = mx.sym.Convolution(data=fpn_p6_1x1,kernel=(3,3), pad = (1,1), stride = (1,1),num_filter = feature_dim, name = 'fpn_p6')
        fpn_p5 = mx.sym.Convolution(data=fpn_p5_plus, kernel=(3,3), pad = (1,1), stride = (1,1),num_filter = feature_dim, name = 'fpn_p5')
        fpn_p4 = mx.sym.Convolution(data=fpn_p4_plus, kernel=(3,3), pad = (1,1), stride = (1,1),num_filter = feature_dim, name = 'fpn_p4')
        fpn_p3 = mx.sym.Convolution(data=fpn_p3_plus, kernel=(3,3), pad = (1,1), stride = (1,1),num_filter = feature_dim, name = 'fpn_p3')
        fpn_p2 = mx.sym.Convolution(data=fpn_p2_plus, kernel=(3,3), pad = (1,1), stride = (1,1),num_filter = feature_dim, name = 'fpn_p2')

        return fpn_p2, fpn_p3,fpn_p4,fpn_p5,fpn_p6

    def get_rpn_subnet(self, feat, num_anchor, suffix):
        rpn_conv = mx.sym.Convolution(data= feat,kernel=(3,3),pad=(1,1),num_filter=512,
                                        weight = self.shared_param_dict['rpn_conv_weight'],
                                        bias = self.shared_param_dict['rpn_conv_bias'],
                                        name = 'rpn_conv_'+suffix)
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type='relu', name = 'rpn_relu_' + suffix)
        rpn_cls_score = mx.sym.Convolution(data = rpn_relu,kernel=(1,1),pad=(0,0),stride=(1,1),num_filter = 2*num_anchor,
                                        weight = self.shared_param_dict['rpn_cls_score_weight'],
                                        bias = self.shared_param_dict['rpn_cls_score_bias'],
                                        name = 'rpn_cls_score_'+suffix)
        
        rpn_bbox_pred = mx.sym.Convolution(data = rpn_relu,kernel=(1,1),pad=(0,0),stride=(1,1),num_filter = 4*num_anchor,
                                        weight = self.shared_param_dict['rpn_bbox_pred_weight'],
                                        bias = self.shared_param_dict['rpn_bbox_pred_bias'],
                                        name = 'rpn_bbox_pred_'+suffix)     
        
        rpn_cls_score_t1 = mx.sym.Reshape(data = rpn_cls_score, shape = (0,2,-1,0),name = 'rpn_cls_score_t1_'+suffix)
        rpn_cls_score_t2 = mx.sym.Reshape(data = rpn_cls_score_t1,shape = (0,2,-1),name = 'rpn_cls_score_t2_'+suffix)
        rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score_t1, mode = 'channel', name = 'rpn_cls_prob_' + suffix)
        rpn_cls_prob_t = mx.sym.Reshape(data=rpn_cls_prob,shape = (0,2*num_anchor, -1, 0), name = 'rpn_cls_prob_t_'+suffix)
        rpn_bbox_pred_t = mx.sym.Reshape(data=rpn_bbox_pred,shape=(0,0,-1),name = 'rpn_bbox_pred_t_'+suffix)

        return rpn_cls_score_t2,rpn_cls_prob_t, rpn_bbox_pred_t, rpn_bbox_pred

    def get_symbol(self, cfg, is_train = True):
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = num_classes
        data = mx.sym.Variable(name = 'data')
        im_info = mx.sym.Variable(name='im_info')

        res2,res3,res4,res5,res6 = self.get_detnet_backbone(data,is_train)
        fpn_p2, fpn_p3,fpn_p4,fpn_p5,fpn_p6 = self.get_fpn_feature(res2, res3, res4, res5,res6)
        #rpn_cls_score_p2:(B,2,(A*W*H))
        #rpn_prob_p2:(B,2A,H,W)
        #rpn_bbox_loss_p2:(B,4A,W*H)
        #rpn_bbox_pred:(B,4A,H,W)
        rpn_cls_score_p2, rpn_prob_p2, rpn_bbox_loss_p2, rpn_bbox_pred_p2 = self.get_rpn_subnet(fpn_p2,cfg.network.NUM_ANCHORS, 'p2')
        rpn_cls_score_p3, rpn_prob_p3, rpn_bbox_loss_p3, rpn_bbox_pred_p3 = self.get_rpn_subnet(fpn_p3,cfg.network.NUM_ANCHORS, 'p3')
        rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4 = self.get_rpn_subnet(fpn_p4,cfg.network.NUM_ANCHORS, 'p4')
        rpn_cls_score_p5, rpn_prob_p5, rpn_bbox_loss_p5, rpn_bbox_pred_p5 = self.get_rpn_subnet(fpn_p5,cfg.network.NUM_ANCHORS, 'p5')
        rpn_cls_score_p6, rpn_prob_p6, rpn_bbox_loss_p6, rpn_bbox_pred_p6 = self.get_rpn_subnet(fpn_p6,cfg.network.NUM_ANCHORS, 'p6')

        #rpn_prob_p2:(1,2A,H,W)
        rpn_cls_prob_dict = {
            'rpn_cls_prob_p6':rpn_prob_p6,
            'rpn_cls_prob_p5':rpn_prob_p5,
            'rpn_cls_prob_p4':rpn_prob_p4,
            'rpn_cls_prob_p3':rpn_prob_p3,
            'rpn_cls_prob_p2':rpn_prob_p2,
        }
        rpn_bbox_pred_dict = {
            'rpn_bbox_pred_p6':rpn_bbox_pred_p6,
            'rpn_bbox_pred_p5':rpn_bbox_pred_p5,
            'rpn_bbox_pred_p4':rpn_bbox_pred_p4,
            'rpn_bbox_pred_p3':rpn_bbox_pred_p3,
            'rpn_bbox_pred_p2':rpn_bbox_pred_p2,
        }
        arg_dict = {**rpn_cls_prob_dict,**rpn_bbox_pred_dict}

        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            #rpn_cls_score:(1,2,(AH1W1+AH2W2+...))
            #rpn_cls_score:(1,4A,(H1W1+H2W2+...))
            rpn_cls_score = mx.sym.Concat(rpn_cls_score_p2,rpn_cls_score_p3,rpn_cls_score_p4,rpn_cls_score_p5,rpn_cls_score_p6,dim = 2)
            rpn_bbox_loss = mx.sym.Concat(rpn_bbox_loss_p2,rpn_bbox_loss_p3,rpn_bbox_loss_p4,rpn_bbox_loss_p5,rpn_bbox_loss_p6,dim = 2)

            rpn_cls_output = mx.sym.SoftmaxOutput(data = rpn_cls_score, label = rpn_label, multi_output= True, normalization='valid',
                                                    use_ignore = True, ignore_label = -1,name='rpn_cls_prob')

            rpn_bbox_loss = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_l1',scalar = 3.0, data=(rpn_bbox_loss - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name = 'rpn_bbox_loss', data = rpn_bbox_loss, grad_scale = 1./cfg.TRAIN.RPN_BATCH_SIZE)

            aux_dict = {
                'op_type': 'pyramid_proposal', 'name':'rois',
                'im_info': im_info,'feat_stride':tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales':tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n':cfg.TRAIN.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n':cfg.TRAIN.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TRAIN.RPN_NMS_THRESH,'rpn_min_size': cfg.TRAIN.RPN_MIN_SIZE
            }
            rois = mx.sym.Custom(**{**arg_dict,**aux_dict})
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes,shape=(-1,5),name = 'gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois = rois, gt_boxes = gt_boxes_reshape,op_type = 'proposal_target',
                                                        num_classes = num_reg_classes, batch_images = cfg.TRAIN.BATCH_IMAGES,
                                                        batch_rois = cfg.TRAIN.BATCH_ROIS, cfg=pickle.dumps(cfg),fg_fraction = cfg.TRAIN.FG_FRACTION)
        
        else:
            aux_dict = {
                'op_type':'pyramid_proposal','name':'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TEST.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TEST.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TEST.RPN_NMS_THRESH, 'rpn_min_size': cfg.TEST.RPN_MIN_SIZE
            }
            rois = mx.sym.Custom(**{**arg_dict,**aux_dict})
        
        roi_pool = mx.sym.Custom(data_p2 = fpn_p2, data_p3 = fpn_p3, data_p4 = fpn_p4, data_p5 = fpn_p5,data_p6 = fpn_p6,
                                 rois = rois, op_type = 'fpn_roi_pooling', name = 'fpn_roi_pooling')
        
        fc_new_1 = mx.sym.FullyConnected(name = 'fc_new_1', data = roi_pool,num_hidden = 1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1,act_type='relu',name = 'fc_new_1_relu')

        fc_new_2 = mx.sym.FullyConnected(name = 'fc_new_2', data = fc_new_1_relu,num_hidden = 1024)
        fc_new_2_relu = mx.sym.Activation(data = fc_new_2, act_type='relu', name = 'fc_new_2_relu')

        cls_score = mx.sym.FullyConnected(name='cls_score', data = fc_new_2_relu,num_hidden = num_classes)
        bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data = fc_new_2_relu,num_hidden = num_reg_classes * 4)

        if is_train:
            cls_prob = mx.sym.SoftmaxOutput(name = 'cls_prob',data=cls_score,label = label, normalization = 'valid')
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name = 'bbox_loss_', scalar = 1.0, data = (bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name = 'bbox_loss', data = bbox_loss_, grad_scale = 1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = label

            rcnn_label = mx.sym.Reshape(data =rcnn_label, shape = (cfg.TRAIN.BATCH_IMAGES, - 1),name = 'label_reshape')
            cls_prob = mx.sym.Reshape(data = cls_prob, shape = (cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name = 'cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape = (cfg.TRAIN.BATCH_IMAGES, -1, 4*num_reg_classes), name = 'bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_output, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name = 'cls_prob', data = cls_score)
            cls_prob = mx.sym.Reshape(data = cls_prob, shape = (cfg.TEST.BATCH_IMAGES,-1,num_classes),name = 'cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data = bbox_pred, shape = (cfg.TEST.BATCH_IMAGES,-1,4*num_reg_classes),name = 'bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
       # group = mx.sym.Group([fpn_p2, fpn_p3,fpn_p4,fpn_p5,fpn_p6])
       # group = mx.sym.Group([rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4])
       # group = mx.sym.Group([rois])
       # self.sym = group

        return group
        


    def init_weight(self, cfg, arg_params, aux_params):
        for name in self.shared_param_list:
            arg_params[name + '_weight'] = mx.random.normal(0,0.01, shape = self.arg_shape_dict[name + '_weight'])
            arg_params[name + '_bias'] = mx.nd.zeros(shape = self.arg_shape_dict[name + '_bias'])
