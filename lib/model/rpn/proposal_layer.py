import torch.nn as nn
import torch
import numpy as np
from .generate_anchors import generate_anchors
from ..utils.config import cfg
from lib.model.nms.nms_wrapper import nms
from .bbox_transform import bbox_transform_inv, clip_boxes

class _ProposalLayer(nn.Module):
    '''
    这个类的作用是
    step1.在feature_map上产生的基础推荐框anchores: [batch_size, K*A, 4]
    step2.与RPN训练过后的基础推荐框的偏移参数 相运算得到RPN网络最后真正的推荐框集合。
    step3.通过将推荐框内有物体的得分排序，按分数从高到低取12000个推荐框
    step4.通过nms将每个feature_map上的推荐框缩小至2000个推荐框，并输出
    (step1. get base recommendation box generated on the feature graph
     step2. After operation of boxes and bbox_deltas，all the anchors of feature_map
            perhaps have the different width，height，center_x and center_y.The result is
            the truely recommendation-boxes of the RPN-net
     step3. By sorting the score of the recommendation box, 12000 recommendation boxes
            are selected according to the scores from high to low.
     step4. By NMS, the num of recommended boxes on each feature_map is reduced to 2000)
    '''
    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        # input[0]: rpn_cls_prob.data [batch_size, 18, H, W]
        # input[1]: rpn_bbox_pred.data [batch_size, 36, H, W]
        # input[2]: im_info [h,w,ratio]
        # input[3]: cfg_key
        scores = input[0][:,self._num_anchors: , :,:]
        bbox_deltas = input[1]
        im_info = input[2]

        cfg_key = input[3]
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N  # 12000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 2000
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # 0.7
        min_size = cfg[cfg_key].RPN_MIN_SIZE  # 8

        batch_size = bbox_deltas.size(0)

        feat_hegiht, feat_width = scores.size(2), scores.size(3)

        #shift_x:[W]->[0, 16, 32, 48...,(W-1)*16]
        shift_x = np.arange(0, feat_width) * self._feat_stride

        #shift_Y:[H]->[0, 16, 32, 48...,(H-1)*16]
        shift_y = np.arange(0, feat_hegiht) * self._feat_stride

        #shift_x:[H, W]->[[0, 16, 32, 48...,(W-1)*16],
        #                 [0, 16, 32, 48...,(W-1)*16],
        #                        ..............       ]
        #shift_y:[H, W]->[[0, 0, 0, 0...]
        #                 [16,16,16,16..]
        #                   ...........
        #                 [(H-1)*16,....]]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        #shifts:[H*W, 4]->[[0,  0,  0, 0],
        #                   ..........
        #                  [(W-1)*16, 0, (W-1)*16, 0],
        #                  [ 0 ,16, 0, 16],
        #                    ............
        #                  [(W-1)*16, 16, (W-1)*16, 16],
        #                     ............
        #                  [(W-1)*16, (H-1)*16, (W-1)*16, (H-1)*16]]
        #
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors #9
        K = shifts.size(0) #feature_map->(H, W) -> H * W = K

        self._anchors = self._anchors.type_as(scores)

        #anchors:[K, A, 4] 《=》[H*W, A, 4]
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K*A, 4).expand(batch_size, K*A, 4)

        #bbox_delta:[batch_size, 36, H, W] => [batch_size, H, W, 36(9 anchors * 4)]
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        #bbox_delta:[batch_size, H, W, 36(9 anchors * 4)] => [batch_size, H*w*9, 4]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        #scores:[batch_size, 9, H, W] => [batch_szie, H, W, 9]
        scores = scores.permute(0, 2, 3, 1).contiguous()
        #scores:[batch_szie, H, W, 9] => [batch_size, H*W*9]
        scores = scores.view(batch_size, -1)

        #1.convert anchors into proposals
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        #2.clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        #1:维度，True代表降序(1:Which dimension is sorted; True:descending order)
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            #3.remove predicted boxes with either height or width < threshold
            proposals_single = proposals_keep[i]
            scores_single = scores[i]
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores.numel():
                order_single = order_single[:pre_nms_topN]


            # proposal_single:[batch_size, pre_nms_topN, 4]
            # scores_single : [batch_size, pre_nms_topN, 1]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0 :
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            #output[i,:,0]是为了区分一个batch中的不同图片，
            #因为这些推荐框是在不同的feature_map上进行后续的选取
            output[i,:,0] = i
            output[i,:num_proposal, 1:] = proposals_single

        return output




