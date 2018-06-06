import torch
import torch.nn as nn
from lib.model.utils.config import cfg
import torch.nn.functional as F
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from torch.autograd import Variable

class _RPN(nn.Module):
    def __init__(self, din):
        super(_RPN, self).__init__()

        self.din = din
        self.anchor_scales = cfg.ANCHOR_SCALES #[8, 16, 32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS #[0.5, 1, 2]
        self.feat_stride = cfg.FEAT_STRIDE[0] #[16]

        #define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
                    input_shape[0],
                    int(d),
                    int(float(input_shape[1] * input_shape[2]) / float(d)),
                    input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        batch_size =base_feat.size(0)

        rpn_conv1 = F.relu(self.RPN_Conv(base_feat),inplace=True)

        #input:[batch_size, 512, H, W]
        #output:[batch_szie, self.nc_score_out, H, W]
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score,2)
        rpn_cls_prob_reshape =F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        cfg_key = 'TRAIN' if self.training else 'TEST'

        # rois：[batch_size, post_nms_topN, 5(1+4)] 1:代表是第几张照片，4代表推荐框参数
        # the fist parameter represents the ROI is in which picture
        # the rest parameter represents the ROI
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        if self.training:
            assert gt_boxes is not None
            # labels:[batch_size,1, A*height, width]
            # bbox_target:[batch_size, A*4, height,width]
            # bbox_inside_weights:[batch_size,A*4,height,width]
            # bbox_outside_weight:[batch_size,A*4,height,width]
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            #print(rpn_data)
            # rpn_cls_score:[batch_size, H*W*9, 2]
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            # label：[batch_size, A*H*W]
            rpn_label = rpn_data[0].view(batch_size, -1)

            # rpn_keep是返回rpn_label中不为-1的位置
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            # index_select是选择rpn_keep所对应比例的行
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)

            # rpn_cls_score为返回256个包含正样本和负样本的在前景与背景上的得分
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box