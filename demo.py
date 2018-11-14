import argparse
import mxnet as mx
import numpy as np
import os
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
from core.tester import Predictor, im_detect
from utils.load_model import load_param
from config.config import config, update_config
from utils.image import transform, resize
from utils.create_logger import create_logger
from nms.nms import py_nms

names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
name_dict = dict(zip(range(80),names))
DEBUG = True
def parse_args():
    parser = argparse.ArgumentParser(description='demo of detnet network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='./cfgs/detnet.yaml', type=str)

    args, rest = parser.parse_known_args()

    parser.add_argument('--thresh', help='valid detection threshold', default=0.5, type=float)
    parser.add_argument('--image', help='image path', default='./images', type=str)
    args = parser.parse_args()
    update_config(args.cfg)
    return args

def demo(cfg, 
              ctx, prefix, epoch,
              has_rpn, thresh,image_path):
    pprint.pprint(cfg)
    data = mx.sym.Variable('data')
    if has_rpn:
        sym_instance = eval(cfg.symbol+'.'+cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train = False)
    else:
        assert False,'does not support'

    arg_params, aux_params = load_param(prefix, epoch)
    
    stride = cfg.network.IMAGE_STRIDE 
    data = {}
    label = None
    predictor = Predictor(sym, ['data','im_info'],None,ctx,None, [('data',(1,3,800,1200)),('im_info',(1,3))],None, arg_params,aux_params)
    for img_path in sorted(os.listdir(image_path)):
      im = cv2.imread(os.path.join(image_path,img_path),cv2.IMREAD_COLOR)[:,:,::-1]
      im_show = im.copy().astype(float)/255
      im = transform(im, cfg.network.PIXEL_MEANS, cfg.network.PIXEL_STDS)
      padded_im, im_scale = resize(im, cfg.SCALES[0][0], cfg.SCALES[0][1], stride = cfg.network.IMAGE_STRIDE)
      padded_im = padded_im.transpose(2,0,1)[None,:,:,:]

      b, c, im_height, im_width = padded_im.shape
      data['data'] = mx.nd.array(padded_im)
      data['im_info'] = mx.nd.array([[im_height, im_width, im_scale]])
      [print(k,v.shape) for k,v in data.items()]
      data_batch = mx.io.DataBatch(data=[data['data'],data['im_info']],label=[],
                                   provide_data = [(k,v.shape) for k,v in data.items()],
                                   provide_label = None)
      
      scores_all, pred_boxes_all, data_dict_all = im_detect(predictor, data_batch, ['data','im_info'],[1],cfg)
    
      num_classes = cfg.dataset.NUM_CLASSES
      
      all_boxes = [[] for _ in range(num_classes)]
      scores = scores_all[0]
      boxes = pred_boxes_all[0]
      cls = scores.argmax(axis = 1)
      cls_scores = scores.max(axis = 1)
      for idx in range(1, num_classes):
        keep_idx = np.where(cls == idx)[0]
        cls_boxes = boxes[keep_idx, idx * 4: (idx + 1) * 4]
        cls_score = cls_scores[keep_idx,np.newaxis]
        cls_dets = np.hstack((cls_boxes, cls_score)).copy()
        all_boxes[idx] = cls_dets
        
      import matplotlib.pyplot as plt
      plt.switch_backend('agg') 
      #im_show = plt.imread(os.path.join(image_path,img_path))
      plt.imshow(im_show)
      for idx in range(1, num_classes):
          keep = py_nms(all_boxes[idx],cfg.TEST.NMS)
          all_boxes[idx] = all_boxes[idx][keep,:]
          for rect in all_boxes[idx]:
            if rect[-1] < 0.3:
                continue
            tl = tuple(rect[:2].astype(np.int64))
            br = tuple(rect[2:4].astype(np.int64))
            color = np.random.rand(3)
            #cv2.rectangle(im,tl,br,color,1)
            #cv2.putText(im,name_dict[idx] + " " + str(int(rect[-1]*100)),tl,cv2.FONT_HERSHEY_SIMPLEX,2,color,1)
            rectangle = plt.Rectangle(tl,br[0] - tl[0],br[1] - tl[1], fill = False,
                                 edgecolor = color,
                                 linewidth = 3.5)
            plt.gca().add_patch(rectangle)
            plt.gca().text(tl[0],tl[1]-2,'{:s} {:.3f}'.format(name_dict[idx], rect[-1]),
                           bbox = dict(facecolor = color, alpha = 0.5),
                           fontsize = 12, color = 'white')
      print("saving image {} in ./det_images".format(img_path))
      plt.savefig("./det_images/{}".format(img_path))
      #cv2.imshow("det_img",im)
      #cv2.waitKey()



    
if __name__ =="__main__":
    args = parse_args()
    
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    ctx = [ctx[0]]
    if DEBUG:
      pass
    print(args)
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    demo(config, 
              ctx,
              os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix),
              config.TEST.test_epoch, 
              config.TEST.HAS_RPN,
              args.thresh,args.image)

