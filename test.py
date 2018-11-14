import argparse
import mxnet as mx
import numpy as np
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
import sys
import cv2
import pprint
import logging
import time
curr_path = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(curr_path,'lib'))
from symbols import detnet
from dataset import *
from core.testloader import TestLoader
from core.tester import Predictor, pred_eval
from utils.load_model import load_param
from config.config import config, update_config

from utils.create_logger import create_logger

DEBUG = False
def parse_args():
    parser = argparse.ArgumentParser(description='Test a detnet network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='./cfgs/detnet.yaml', type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', default=True)
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger = None, output_path = None):
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))
    if has_rpn:
        sym_instance = eval(cfg.symbol+'.'+cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train = False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        assert False,'do not support'

    test_data = TestLoader(roidb,cfg,batch_size = len(ctx),shuffle = shuffle, has_rpn = has_rpn)
#    if DEBUG:
#      test_data.reset()
#      print(test_data.provide_data)
#      print(test_data.provide_label)
#      info, it = test_data.next()
#      print(it)
#      print(info)
#      print(info.data[0])
#      print(info.label[0])
#      print(info.provide_data)
#      print(info.provide_label)



    arg_params, aux_params = load_param(prefix, epoch, process = True)

    data_shape_dict = dict(test_data.provide_data)
    sym_instance.infer_shape(data_shape_dict)
    data_names = [k[0] for k in test_data.provide_data]
    label_names = None
    max_data_shape = [('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([int(v[1]//16*16) for v in cfg.SCALES])))]
    predictor = Predictor(sym, data_names, label_names,
                          context = ctx, max_data_shapes = max_data_shape,
                          provide_data = test_data.provide_data,provide_label = test_data.provide_label,
                          arg_params = arg_params, aux_params = aux_params)
    
    pred_eval(predictor, test_data, imdb, cfg, vis = vis, ignore_cache = ignore_cache, thresh = thresh, logger = logger)
if __name__ =="__main__":
    args = parse_args()
    
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print(args)
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    test_rcnn(config, config.dataset.dataset,config.dataset.test_image_set, config.dataset.root_path,
              config.dataset.dataset_path, ctx,
              os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix),
              config.TEST.test_epoch, args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal,
              args.thresh,logger = logger,output_path = final_output_path)

