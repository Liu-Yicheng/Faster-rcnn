import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data.sampler import Sampler
from lib.model.utils.config import cfg
from scipy.misc import imread
import cv2

class sampler(Sampler):
    '''
    这个采样类是根据训练集总数量与batch-size来划分index，形成index的集合
    例如：step1. train_size = 6, batch_size = 2 =>数据集的索引集合Index为[0,1,2,3,4,5]
         step2. 将Indexs随机打乱并且分成 train_size/batch_size 个集合：[1,3],[0,4],[2,5]
         step3. 每次迭代时,dataloader会调用该类的iter方法拿到一个index的集合--：[1,3]

    This sampler is used to form a set of index according to train_size and batch_size
    example:
         step1. train_size = 6, batch_size=2 => Index is [0,1,2,3,4,5]
         step2. Index is shuffled and divided into some small sets which number is train_size/batch_size
                [1,3],[0,4],[2,5]
    '''
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size :
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size)
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size

        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


class single_data_Loader(data.Dataset):
    '''
    该类的作用主要是获取单个图片信息
    输入： roidb--处理过后的数据集
          roidb_list--数据集中的图片的长宽比按大小顺序排列
          ratio_index--roidb_list的对应位置图片的index
    step1.根据数据集总量和batch_size将数据集划分为num_batch个batch
          每个batch内的图片长宽比要设置成一样
    step2.dataloader调用时，会调用该类的gettem方法
          根据step1确定的目标长宽比，对图片进行裁剪或填充，使它的原始长宽比变为目标长宽比
          裁剪或填充取决于['crop']
    '''
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes

        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_HEIGHT
        self.max_num_box = cfg.MAX_NUM_GT_BOXES

        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int (np.ceil(len(ratio_index) / batch_size))

        for i in range(num_batch):
            left_index = i * batch_size
            right_index = min((i+1)*batch_size-1, self.data_size-1)
            if ratio_list[right_index] < 1:
                target_ratio = ratio_list[left_index]
            elif ratio_list[left_index] > 1:
                target_ratio = ratio_list[right_index]
            else:
                target_ratio = 1
            self.ratio_list_batch[left_index: (right_index+1)] = target_ratio

    def __getitem__(self, index):
        if self.training :
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        minibatch_db = [self._roidb[index_ratio]]
        blobs = self.get_minibatch(minibatch_db, self._num_classes)

        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])

        data_height, data_width = data.size(1), data.size(2)
        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            ratio = self.ratio_list_batch[index]
            if self._roidb[index_ratio]['need_crop']:
                if ratio < 1:
                    min_y = int(torch.min(gt_boxes[:,1]))
                    max_y = int(torch.max(gt_boxes[:,3]))
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height

                    boxes_region = max_y -min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        if (boxes_region - trim_size) < 0:
                            y_s_min = max(max_y-trim_size, 0)
                            y_s_max = min(min_y, data_height-trim_size)
                            if y_s_min == y_s_max :
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((boxes_region - trim_size)/2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y+y_s_add))
                    data = data[:, y_s:(y_s+trim_size),:,:]

                    gt_boxes[:, 1] = gt_boxes[:,1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:,3] - float(y_s)

                    gt_boxes[:, 1].clamp_(0, trim_size-1)
                    gt_boxes[:, 3].clamp_(0, trim_size-1)

                else:

                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region - trim_size) < 0:
                            x_s_min = max(max_x - trim_size, 0)
                            x_s_max = min(min_x, data_width - trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region - trim_size) / 2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x + x_s_add))

                    data = data[:, :, x_s:(x_s + trim_size), :]


                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)

                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            if ratio < 1:
                trim_size = int(np.floor(data_width / ratio))
                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), data_width, 3).zero_()
                padding_data[:data_height, :, :] = data[0]
                im_info[0, 0] = padding_data.size(0)
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.FloatTensor(data_height, \
                                                 int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                # gt_boxes.clamp_(0, trim_size)
                gt_boxes[:, :4].clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size

            not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
            keep = torch.nonzero(not_keep ==0).view(-1)
            gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)

                gt_boxes_padding[:num_boxes,:]=gt_boxes[:num_boxes,:]
            else:
                num_boxes = 0

            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)

            return padding_data, im_info, gt_boxes_padding, num_boxes

        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            im_info = im_info.view(3)

            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0

            return data, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self._roidb)


    def get_minibatch(self, roidb, num_classes):
        num_images= len(roidb)
        random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES),size = num_images)

        assert (cfg.TRAIN.BATCH_SIZE % num_images ==0),\
               'num images ({}) must divide Batch Size ({})'.\
                format(num_images, cfg.TRAIN.BATCH_SIZE)

        im_blob, im_scales = self._get_image_blob(roidb, random_scale_inds)
        blobs = {'data': im_blob}
        assert len(im_scales) ==1, 'Single batch only'
        assert len(roidb) ==1, 'Single batch only'

        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] =roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        blobs['img_id'] = roidb[0]['img_id']
        return blobs

    def  _get_image_blob(self, roidb, scales_inds):
        num_images = len(roidb)
        processed_ims = []
        im_scales =[]
        for i in range(num_images):
            #print(roidb[i])
            im = imread(roidb[i]['image'])
            if len(im.shape) == 2:
                im = im[:,:,np.newaxis]
                im = np.concatenate((im, im, im), axis=2)
            #rgb->bgr
            im = im[:,:,::-1]
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = cfg.TRAIN.SCALES[scales_inds[i]]
            im, im_scale = self.prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            im_scales.append(im_scale)
            processed_ims.append(im)
        blob = self.im_list_to_blob(processed_ims)
        return blob, im_scales

    def prep_im_for_blob(self, im, pixel_means, target_size, max_size):
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        return im, im_scale

    def im_list_to_blob(self, ims):
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        return blob
