from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import torch
from torch.autograd import Variable
from Data.pascal import get_imdb_and_roidb
from Data.batch_loader import sampler, single_data_Loader
from tqdm import  tqdm
from torch.utils.data.sampler import Sampler

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.faster_rcnn.vgg16 import vgg16

def parse_args():
  """
  通常需要修改的几个参数：
    --epochs：共需要训练几遍数据集
    --save_epoch：每训练几遍数据集就保存训练模型
    --cuda：根据电脑上有没有gpu决定,但一般没有gpu跑不了这个程序
    --bs：batch_size大小
    --lr_decay_step：每训练几遍数据集,使学习率衰减
    --lr_decay_gamma：学习率衰减的比例

  Several parameters that usually need to be modified：
    --epoch： You will train your dataset epoch-times
    --save_epoch： Save model when you train your dataset save_epoch-taimes
    --cada： It depends on your computer‘s gpu, you can't run this code without gpu
    --bs: Batch-size
    --lr_decay_step: Learning-rate needs to be attenuated when the dataset is trained lr_dacay_step times
    --lr_decay_gamma: The ratio of the attenuation of learning rate
  """
  parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')

  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=50, type=int)
  parser.add_argument('--save_epochs', dest='save_epochs',
                      help='save_epochs',
                      default=50, type=int)

  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=1, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)


  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)

  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True, type=bool)

  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      

  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      default=True, type=bool)

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=10, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.5, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

  args = parser.parse_args()
  return args



if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  #导入VGG16模型训练的一些参数(import some parameters for VGG16 model)
  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  args.cfg_file =  "cfgs/vgg16.yml"
  cfg_from_file(args.cfg_file)
  cfg_from_list(args.set_cfgs)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  #图片增强：数据集中的图片进行水平反转来增加图片数量
  #Image enhancement: images in datasets are horizontally reversed to increase number of picture
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda

  output_dir = './Output'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  #这个部分是数据获取部分,包括：处理原始数据、得到采样类、得到单个图片信息获取类、构造dataloader
  #This part is the data-acquisition part, including：1.processing raw picture, 2.get sampler
  #3.get the Class which is for getting imformation of single picture  4.get the dataloader
  imdb, roidb, ratio_list, ratio_index = get_imdb_and_roidb('train')#'train'代表读取的是train.txt文件
  train_size = len(roidb)
  sampler_batch = sampler(train_size, args.batch_size)
  dataset = single_data_Loader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_batch, num_workers=4)
  print('{:d} roidb entries'.format(len(roidb)))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  args.cuda = True

  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr

  #模型内权重与偏差的设定（setting of model’s weight and bias）
  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  #optimizer-setting
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)
  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  #加载自己训练过的模型，用于继续训练
  #load the user-trained model to continue training
  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.cuda:
    fasterRCNN.cuda()

  step = 0
  iters_per_epoch = int(train_size / args.batch_size)

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()
    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    bar = tqdm(dataloader, total = len(dataloader))
    for step,data in enumerate(bar):
      step += 1
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
             + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.data[0]


      optimizer.zero_grad()
      loss.backward()

      clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= args.disp_interval

        loss_rpn_cls = rpn_loss_cls.data[0]
        loss_rpn_box = rpn_loss_box.data[0]
        loss_rcnn_cls = RCNN_loss_cls.data[0]
        loss_rcnn_box = RCNN_loss_bbox.data[0]
        fg_cnt = torch.sum(rois_label.data.ne(0))
        bg_cnt = rois_label.data.numel() - fg_cnt
        bar.set_description("epoch{:2d} lr:{:.2e} loss:{:.4f} :rpn_cls:{:.4f},rpn_box:{:.4f},rcnn_cls:{:.4f},rcnn_box{:.4f}" \
                                .format(epoch, lr, loss_temp,loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        loss_temp = 0
        start = time.time()
    if epoch % args.save_epochs ==0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}.pth'.format(epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
          }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
