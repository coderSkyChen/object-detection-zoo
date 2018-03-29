# @Time    : 2018/3/27 8:37
# @File    : opts.py.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

import argparse
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=5, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./models",
                        nargs=argparse.REMAINDER)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='model')
    parser.add_argument('--root_output', type=str, default='output')
    # set training session
    parser.add_argument('--train_id',
                        help='train_id is used to identify this training phrase', type=str)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')

    args = parser.parse_args()
    args.imdb_name = "voc_2007_trainval"
    args.imdbval_name = "voc_2007_test"
    args.ANCHOR_SCALES = [8, 16, 32]
    args.ANCHOR_RATIOS = [0.5, 1, 2]
    args.MAX_NUM_GT_BOXES = 20
    args.BATCH_SIZE = 256
    args.EXP_DIR = 'vgg16'
    args.HAS_RPN = True
    args.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    args.RPN_POSITIVE_OVERLAP = 0.7
    args.RPN_NEGATIVE_OVERLAP = 0.3
    args.RPN_CLOBBER_POSITIVES = False
    args.RPN_FG_FRACTION = 0.5
    args.RPN_NMS_THRESH = 0.7
    args.RPN_PRE_NMS_TOP_N = 12000
    args.RPN_POST_NMS_TOP_N = 2000
    args.RPN_MIN_SIZE = 8
    args.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    args.RPN_POSITIVE_WEIGHT = -1.0
    args.USE_ALL_GT = True
    args.RPN_BATCHSIZE = 256
    args.PROPOSAL_METHOD = 'gt'
    args.BG_THRESH_LO = 0.0
    args.POOLING_MODE = 'align'
    args.CROP_RESIZE_WITH_MAX_POOL = False
    args.RNG_SEED = 3
    args.MOMENTUM = 0.9
    args.WEIGHT_DECAY = 0.0005
    args.GAMMA = 0.1
    args.STEPSIZE = [30000]
    args.DISPLAY = 10
    args.DOUBLE_BIAS = True
    args.TRUNCATED = False
    args.BIAS_DECAY = False
    args.USE_GT = False
    args.ASPECT_GROUPING = False
    args.SNAPSHOT_KEPT = 3
    args.SCALES = (600,)
    args.MAX_SIZE = 1000
    args.TRIM_HEIGHT = 600
    args.TRIM_WIDTH = 600
    args.IMS_PER_BATCH = 1
    args.FG_FRACTION = 0.25
    args.FG_THRESH = 0.5
    args.BG_THRESH_HI = 0.5
    args.BG_THRESH_LO = 0.1
    args.USE_FLIPPED = True
    args.BBOX_THRESH = 0.5
    args.BBOX_NORMALIZE_TARGETS = True
    args.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    args.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    args.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
    args.DEDUP_BOXES = 1. / 16.
    args.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    args.EPS = 1e-14
    args.DATA_DIR = '/home/mcg/cxk/dataset'
    args.USE_GPU_NMS = True
    args.POOLING_SIZE = 7
    args.FEAT_STRIDE = [16, ]
    args.CROP_RESIZE_WITH_MAX_POOL = True
    args.TEST_NMS = 0.3
    args.TEST_SCALES = (600,)
    args.TEST_MAX_SIZE = 1000
    args.TEST_BBOX_REG = True
    return args
