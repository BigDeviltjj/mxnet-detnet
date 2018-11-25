# DetNet : A Backbone network for Object Detection

This repository tries to reproduce the result in [DetNet: A Backbone network for Object Detection](https://arxiv.org/pdf/1804.06215.pdf) using mxnet.

DetNet is evolved from fpn, thus one can change this repository to fpn easily.

### demo result

![demo](https://github.com/BigDeviltjj/mxnet-detnet/blob/master/det_images/000000124442.jpg)

### set up environment

1.clone this repository into the directory.

```
git clone git https://github.com/BigDeviltjj/mxnet-detnet.git
```

2.download coco dataset into data directory.

3.run`sh init.sh`.

4.you may need to install python modules including cv2, matplotlib, numpy etc.

## mAP
|        Model          | Training data    | Test data |  mAP |
|:-----------------:|:----------------:|:---------:|:----:|
| [detnet_coco-0016.params](https://drive.google.com/drive/folders/1Aon21uVFEsWTzDG2m1AMuo8ka-YZlRS3)(multi_scale)| train2017| val2017| 39.7|

### demo

Put the original images in images/, then run

```
python demo.py
```

detected images will be saved in det_images/


### train the model

You need to put the coco2017 dataset in data/ folder, and download pretrained model from [detnet-0000.params](https://drive.google.com/drive/folders/1Aon21uVFEsWTzDG2m1AMuo8ka-YZlRS3). Then put them in folder model/pretrained_mode, then run

```
python train_end2end.py
```

### evaluate the model

Download the compressed trained model and symbol file [detnet_coco-0016.params](https://drive.google.com/drive/folders/1Aon21uVFEsWTzDG2m1AMuo8ka-YZlRS3) and unzip them then put them in folder output/detnet/coco/detnet/train2017/, then run

```
python test.py
```
