# DetNet in mxnet

This repository tries to reproduce the result in [DetNet: A Backbone network for Object Detection](https://arxiv.org/pdf/1804.06215.pdf).

### set up environment

1.clone this repository into the directory.

```
git clone git https://github.com/BigDeviltjj/mxnet-detnet.git
```

2.download coco dataset into data directory.

3.run`sh init.sh`.

4.specific requirements can be obtained from error message when running the program.

5.please make sure to clone the lastest commit version of this repository

### demo

```
python demo.py
```

### train the model

```
python train_end2end.py
```

### evaluate the model

```
python test.py
```

### TODO

* offering pretained model

* synchronized batchnorm layer for multi-gpu training

* replacing roi pooling layer with roi align layer


