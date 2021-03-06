# @Time    : 2018/3/28 13:28
# @File    : demo.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from scipy.misc import imread
from model import clip_boxes
from nms.nms_wrapper import nms
from model import bbox_transform_inv
from model import save_net, load_net, vis_detections
from model import vgg16
from opts import parse_args

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """

    def im_list_to_blob(ims):
        """Convert a list of images into a network input.
        Assumes images are already prepared by prep_im_for_blob(means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= args.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in args.TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > args.TEST_MAX_SIZE:
            im_scale = float(args.TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    np.random.seed(args.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = os.path.join('./data/results', args.train_id, 'model', args.model_name)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)

    pascal_classes = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic, args=args)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (input_dir))
    checkpoint = torch.load(input_dir)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        args.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if args.cuda > 0:
        args.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        imglist = os.listdir(args.image_dir)
        num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    while (num_images >= 0):
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            # im = cv2.imread(im_file)
            im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if args.TEST_BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if args.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(args.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(args.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(args.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(args.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, args.TEST_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()

        if vis and webcam_num == -1:
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()