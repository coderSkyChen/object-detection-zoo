# object-detection-zoo
Codes for popular object detection models，verified on the Pascal VOC 2007 dataset.

# Faster-rcnn

This code can be seen as a **simplified** version for [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), it took me half a week to go through @jwyang 's code, their code is valuable but it's too robust to be easily understood for the ones that are unfamiliar with object detection task. So this code is released for easy understanding and it's build on top of **faster-rcnn.pytorch**.

**Main Reference Paper**: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

## Depencies
- Pytorch-0.2.0_3
- Tensorflow-1.3.1，this is only for the using of tensorboard, it's ok without this but you need to comment the corresponding codes, because i'm too lazy to make it aotumatically. 0.0.
- CUDA 7.5 or higher

## Data preparation
Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, modify the `args.DATA_DIR` in opts.py.

## Pretrained Model
I used VGG-16 Caffe pretrained models(pretrained on ImageNet), you can download it from:[Baidu Cloud](https://pan.baidu.com/s/1wN1wVeYQx6DHN0OaXxCBGg)

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

## Checkpoint Model
This model is the whole model that has been trained by myself, test with it you'll get mAP score:78.4 on PASCAL VOC test set.

You can download it from:[Baidu Cloud]()

## Compilation
```
cd lib
sh make.sh
```
"It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version."
If you encounterd any issues, please refer to [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

## Scripts for training, testing and demo
```
Train:
CUDA_VISIBLE_DEVICES=0 python trainval.py --dataset pascal_voc --net vgg16 --bs 8 --nw 32 --cuda --train_id 0.1

Test:
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pascal_voc --net vgg16 --bs 1 --cuda --train_id 0.1 --model_name faster_rcnn_19_1251.pth

Demo:
# before run the code, you need to add images into ./images folder.
CUDA_VISIBLE_DEVICES=0 python demo.py --net vgg16 --cuda --train_id 0.1 --model_name faster_rcnn_19_1251.pth
```
## Benchmarking

PASCAL VOC 2007 (Train/Test: trainval/test,scale=600, ROI Align)

|model|#GPUs|batch size|lr|lr_decay|max_epoch|mAP|
|--------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|VGG-16|1|8|0.001|5|20|78.4|

## Demo
![3](https://github.com/coderSkyChen/object-detection-zoo/raw/master/images/3_det.jpg)
![5](https://github.com/coderSkyChen/object-detection-zoo/raw/master/images/5_det.jpg)
