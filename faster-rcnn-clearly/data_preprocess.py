# @Time    : 2018/4/20 15:34
# @File    : data_preprocess.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

import os
import uuid
import pickle
import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
import scipy.sparse
import PIL
from voc_eval import voc_eval
import subprocess


class PascalData:
    def __init__(self, image_set, args, devkit_path=None):
        self.args = args
        self._image_set = image_set
        self._devkit_path = os.path.join(self.args.DATA_DIR, 'VOCdevkit' + '2007')
        self._data_path = os.path.join(self._devkit_path, 'VOC2007')
        self.classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')        
        self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self._image_ext = '.jpg'
        self._image_index = self._get_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

        self.roidb = self.gt_roidb()

        if args.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            self.append_flipped_images()
            print('done')

        print('Enriching training data...')
        sizes = [PIL.Image.open(os.path.join(self._data_path, 'JPEGImages',\
                    self._image_index[i]+self._image_ext)).size
                 for i in range(len(self._image_index))]

        for i in range(len(self._image_index)):
            self.roidb[i]['img_id'] = i
            self.roidb[i]['image'] = os.path.join(self._data_path, 'JPEGImages',\
                    self._image_index[i]+self._image_ext)
            self.roidb[i]['width'] = sizes[i][0]
            self.roidb[i]['height'] = sizes[i][1]
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = self.roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            maxclasses = gt_overlaps.argmax(axis=1)
            self.roidb[i]['maxclasses'] = maxclasses
            self.roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(maxclasses[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(maxclasses[nonzero_inds] != 0)
        print('done')

        # filter the image without bounding box.
        print('before filtering, there are %d images...' % (len(self.roidb)))
        i = 0
        while i < len(self.roidb):
            if len(self.roidb[i]['boxes']) == 0:
                del self.roidb[i]
                i -= 1
            i += 1

        print('after filtering, there are %d images...' % (len(self.roidb)))

        self.ratio_list, self.ratio_index = self.rank_roidb_ratio()


    def rank_roidb_ratio(self):
        # rank roidb based on the ratio between width and height.
        ratio_large = 2  # largest ratio to preserve.
        ratio_small = 0.5  # smallest ratio to preserve.

        ratio_list = []
        for i in range(len(self.roidb)):
            width = self.roidb[i]['width']
            height = self.roidb[i]['height']
            ratio = width / float(height)

            if ratio > ratio_large:
                self.roidb[i]['need_crop'] = 1
                ratio = ratio_large
            elif ratio < ratio_small:
                self.roidb[i]['need_crop'] = 1
                ratio = ratio_small
            else:
                self.roidb[i]['need_crop'] = 0

            ratio_list.append(ratio)

        ratio_list = np.array(ratio_list)
        ratio_index = np.argsort(ratio_list)
        return ratio_list[ratio_index], ratio_index

    def image_path_at(self, i):
        return os.path.join(self._data_path, 'JPEGImages', \
                     self._image_index[i] + self._image_ext)
    def append_flipped_images(self):
        num_images = len(self._image_index)
        widths = [PIL.Image.open(os.path.join(self._data_path, 'JPEGImages',\
                    self._image_index[i]+self._image_ext)).size[0] \
                  for i in range(num_images)]
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gtclasses': self.roidb[i]['gtclasses'],
                     'flipped': True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def _get_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        def load_annotation_for_one_image(index):
            """
            Load image and bounding boxes info from XML file in the PASCAL VOC
            format.
            """
            filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
            tree = ET.parse(filename)
            objs = tree.findall('object')

            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gtclasses = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, len(self.classes)), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)
            ishards = np.zeros((num_objs), dtype=np.int32)

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                diffc = obj.find('difficult')
                difficult = 0 if diffc == None else int(diffc.text)
                ishards[ix] = difficult

                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gtclasses[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            overlaps = scipy.sparse.csr_matrix(overlaps)

            return {'boxes': boxes,
                    'gtclasses': gtclasses,
                    'gt_ishard': ishards,
                    'gt_overlaps': overlaps,
                    'flipped': False,
                    'seg_areas': seg_areas}
        print('Loading roidb...')
        cache_path = './cache'
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        cache_file = os.path.join(cache_path, 'voc_2007_%s_gt_roidb.pkl' % self._image_set)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = pickle.load(f)
            print('gt roidb loaded from {}'.format(cache_file))
            return roidb

        roidb = [load_annotation_for_one_image(index)\
                 for index in self._image_index]
        with open(cache_file, 'wb') as f:
            pickle.dump(roidb, f, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return roidb


    # -----------
    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC2007', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + '2007',
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + '2007',
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int('2007') < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(self.args.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(self.args.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self.classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True