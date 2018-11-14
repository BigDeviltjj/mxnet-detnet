import argparse
import mxnet as mx
import numpy as np
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import sys
import re
import shutil
import pprint
from config.config import config, update_config
curr_path = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'lib'))
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from symbols import detnet
from core.loader import PyramidAnchorIterator
from core import metric
DEBUG = False
def parse_args():
    parser = argparse.ArgumentParser(description='train detnet network')
    parser.add_argument('--cfg',help='configure file name',type = str, default = './cfgs/detnet.yaml')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    shutil.copy2(os.path.join(curr_path, 'symbols',config.symbol+'.py'),final_output_path)
    sym_instance = detnet.detnet()
    sym = sym_instance.get_symbol(config, is_train = True)


    feat_pyramid_level = np.log2(config.network.RPN_FEAT_STRIDE).astype(int)
    feat_sym = [sym.get_internals()['rpn_cls_score_p'+str(x)+'_output'] for x in feat_pyramid_level]

    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_gt_roidb(config.dataset.dataset,image_set, config.dataset.root_path, config.dataset.dataset_path,flip = config.TRAIN.FLIP) for image_set in image_sets]
        
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb,config)
    
    train_data = PyramidAnchorIterator(feat_sym, roidb, config,batch_size = input_batch_size, shuffle = config.TRAIN.SHUFFLE,ctx = ctx, feat_strides = config.network.RPN_FEAT_STRIDE, anchor_scales = config.network.ANCHOR_SCALES,anchor_ratios = config.network.ANCHOR_RATIOS, aspect_grouping = config.TRAIN.ASPECT_GROUPING,allowed_border = np.inf)

    max_data_shape = [('data',(config.TRAIN.BATCH_IMAGES,3,max([v[0] for v in config.SCALES]),max([int(v[1]//16*16) for v in config.SCALES])))]
    max_data_shape,max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes',(config.TRAIN.BATCH_IMAGES,100,5)))
    print('providing maximum shape', max_data_shape, max_label_shape)

    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    fixed_param_names = None
    if config.TRAIN.RESUME:
        print('continue training from ',begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert = True)
#        _, arg_params, aux_params = mx.model.load_checkpoint(prefix,begin_epoch)
    else:
        print('loading pretrained model from {}'.format(pretrained+'_'+str(begin_epoch)))
        arg_params, aux_params = load_param(pretrained, begin_epoch, convert = True)
        sym_instance.init_weight(config, arg_params, aux_params)
        fixed_param_names = list()

        if config.network.FIXED_PARAMS is not None:
            names = sym.list_arguments()
            for name in names:
                for pre in config.network.FIXED_PARAMS:
                    if pre in name:
                        fixed_param_names.append(name)
                        break

        #arg_params, aux_params = None, None
        #sym_instance.init_weight(config, arg_params, aux_params)

    #sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = mx.mod.Module(sym,data_names = data_names,label_names = label_names,
                 logger = logger, context = ctx, fixed_param_names = fixed_param_names)




    rpn_eval_metric = metric.RPNAccMetric()
    rpn_cls_metric = metric.RPNLogLossMetric()
    rpn_bbox_metric = metric.RPNL1LossMetric()
    rpn_fg_metric = metric.RPNFGFraction(config)
    eval_metric = metric.RCNNAccMetric(config)
    eval_fg_metric = metric.RCNNFGAccuracy(config)
    cls_metric = metric.RCNNLogLossMetric(config)
    bbox_metric = metric.RCNNL1LossMetric(config)
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    batch_end_callback = [mx.callback.Speedometer(train_data.batch_size,frequent = 20,auto_reset = False)]
    epoch_end_callback = [mx.callback.do_checkpoint(prefix, period = 1)]
    base_lr = lr 
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor **(len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb)/batch_size) for epoch in lr_epoch_diff]
    print('lr',lr,'lr_epoch_diff',lr_epoch_diff,'lr_iters',lr_iters)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step = lr_iters,factor = lr_factor)
    optimizer_params = {"momentum":config.TRAIN.momentum,
                        'wd':config.TRAIN.wd,
                        'learning_rate':lr,
                        'lr_scheduler':lr_scheduler,
                        'clip_gradient':None}
        
    if not isinstance(train_data,mx.io.PrefetchingIter):
        train_data = mx.io.PrefetchingIter(train_data)

    mod.fit(train_data,eval_metric = eval_metrics,epoch_end_callback=epoch_end_callback,
            batch_end_callback = batch_end_callback, 
            optimizer = 'sgd',optimizer_params = optimizer_params,
            begin_epoch = begin_epoch,
	    arg_params = arg_params,
            aux_params = aux_params,
            num_epoch = end_epoch,
            allow_missing = True)
def main():
    args = parse_args()
    print('called with argument:',args)
    print(args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, 
              config.TRAIN.model_prefix,config.TRAIN.begin_epoch, config.TRAIN.end_epoch,
              config.TRAIN.lr, config.TRAIN.lr_step)
if __name__ == '__main__':
    main()
