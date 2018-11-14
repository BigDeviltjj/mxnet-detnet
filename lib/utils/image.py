from PIL import Image
import numpy as np
import os
import cv2
import random
from bbox.bbox_transform import clip_boxes

def resize(im,target_size, max_size, stride = 0,interpolation = cv2.INTER_LINEAR):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size)/float(im_size_max)
    im = cv2.resize(im,None,None,fx = im_scale,fy = im_scale,interpolation = interpolation)
    if stride == 0:
        return im,im_scale
    else:
        im_height = int(np.ceil(im.shape[0]/float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height,im_width,im_channel))
        padded_im[:im.shape[0],:im.shape[1],:] = im
        return padded_im, im_scale
    
def transform(im, pixel_means, pixel_stds):
    img = im[:,:,::-1]
    img = (img/255 - np.array([[[0.485, 0.456, 0.406]]]))/np.array([[[0.229, 0.224, 0.225]]])

    return im
def get_image(roidb,config):
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:,::-1,:]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im = transform(im,config.network.PIXEL_MEANS, config.network.PIXEL_STDS)
        im_tensor,im_scale = resize(im,target_size, max_size, stride = config.network.IMAGE_STRIDE)
        im_tensor = im_tensor.transpose(2,0,1)[None,:,:,:]
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2],im_tensor.shape[3],im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale),im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

