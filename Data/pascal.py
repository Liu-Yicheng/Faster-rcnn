import os
import pickle
import  xml.etree.ElementTree as ET
import numpy as np
from lib.model.utils.config import cfg
from PIL import Image
def get_imdb_and_roidb(imdb_set, training=True):
    def prepare_roidb(imdb):
        roidb = imdb.roidb
        sizes = [Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
        for i in range(len(imdb.image_index)):
            roidb[i]['img_id'] = i
            roidb[i]['image'] = imdb.image_path_at(i)
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]

            gt_overlaps =roidb[i]['gt_overlaps']#.toarray()

            max_overlaps = gt_overlaps.max(axis=1)
            max_classes = gt_overlaps.argmax(axis = 1)
            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps

            zeros_inds = np.where(max_overlaps == 0 )[0]
            assert all(max_classes[zeros_inds] == 0)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] !=0)

    def get_training_roidb(imdb):
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples')
            imdb.append_flipped_images()
            print('done')
        print('Preparing training data')
        prepare_roidb(imdb)
        print('done')
        return imdb.roidb

    def get_roidb(imdb_set):
        imdb = pascal(imdb_set)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)

        return roidb

    def filter_roidb(roidb):

        print('before filtering, there are {} images...'.format(len(roidb)))
        i = 0
        while i < len(roidb) :
            if len(roidb[i]['boxes']) ==0:

                del roidb[i]
                i-=1
            i+=1
        print('after filtering, there are {} images...'.format(len(roidb)))
        return roidb

    def rank_roidb_ratio(roidb):
        ratio_large = 2
        ratio_small = 0.5

        ratio_list = []
        for i in range(len(roidb)) :
            width = roidb[i]['width']
            height = roidb[i]['height']
            ratio = width / float(height)
            if ratio > ratio_large :
                roidb[i]['need_crop'] = 1
                ratio =ratio_large
            elif ratio < ratio_small :
                ratio[i]['need_crop'] = 1
                ratio = ratio_small
            else:
                roidb[i]['need_crop'] = 0
            ratio_list.append(ratio)
        ratio_list = np.array(ratio_list)
        ratio_index = np.argsort(ratio_list)
        return ratio_list[ratio_index], ratio_index

    roidb = get_roidb(imdb_set)

    imdb =pascal(imdb_set)
    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index



class pascal(object):
    def __init__(self, image_set, data_path=None):
        self._image_set = image_set
        self._data_path = './Data/picture_data'
        self._classes = ('__background__','jyz', 'xjyz', 'dwq', 'xj','cls', 'tgsr', 'xzsr', 'xlxj')

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        self._roidb = None
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._data_path),\
            'Data path does not exist:{}'.format(self._data_path)


    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(self._data_path, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def num_images(self):
        return len(self._image_index)

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self._roidb_handler = method

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, self._image_set+'.txt')
        #image_set_file =os.path.abspath(cfg.train_txt)
        #print(os.path.isfile(image_set_file))
        assert os.path.isfile(image_set_file), \
               'Path does not exist:{}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, 'gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = pickle.load(f)
                print('gt roidb loaded from {}'.format(cache_file))
                return roidb
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self._image_index]
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_roidb, f, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_pascal_annotation(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index+'.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_obj = len(objs)

        boxes = np.zeros((num_obj, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_obj), dtype=np.uint32)
        overlaps = np.zeros((num_obj, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')

            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        return {
                 'boxes':boxes,
                 'gt_classes':gt_classes,
                 'gt_overlaps': overlaps,
                 'flipped': False
               }
    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index+self._image_ext)
        assert os.path.exists(image_path),\
               'Path does not exist {}'.format(image_path)
        return image_path

    def image_path_at(self,i):
        return self.image_path_from_index(self._image_index[i])

    def _get_widths(self):
        return [Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 -1
            boxes[:, 2] = widths[i] - oldx1 -1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = { 'boxes' : boxes,
                      'gt_overlaps' :self.roidb[i]['gt_overlaps'],
                      'gt_classes': self.roidb[i]['gt_classes'],
                      'flipped': True
                    }
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

