import numpy as np
import torch
def generate_anchors(base_size = 16, ratios =[0.5, 1, 2],
                     scales = 2**np.arange(3,6)):
    '''
    输入：参考anchor，ratios，scales
    输出：9个anchor：
             array([[ -83.,  -39.,  100.,   56.],
                    [-175.,  -87.,  192.,  104.],
                    [-359., -183.,  376.,  200.],
                    [ -55.,  -55.,   72.,   72.],
                    [-119., -119.,  136.,  136.],
                    [-247., -247.,  264.,  264.],
                    [ -35.,  -79.,   52.,   96.],
                    [ -79., -167.,   96.,  184.],
                    [-167., -343.,  184.,  360.]])
    (input:reference anchor, ratios, scales
     output:a set of nine anchors)
    '''

    #构造一个基础的anchor，面积为16*16
    # （generate a reference anchor which area is 16*16）
    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    #ratio_anchors: [[-3.5  2.  18.5 13. ]
    #                [ 0.   0.  15.  15. ]
    #                [ 2.5 -3.  12.5 18. ]]
    ratio_anchors = _ratio_enum(base_anchor, ratios)

    anchors = np.vstack([_scale_enum(ratio_anchors[i,:], scales)
                        for i in range(ratio_anchors.shape[0])])

    return anchors

def _whctrs(anchor):
    '''
    输入的是一个anchor的[xmin, ymin, xmax, ymax]
    输出的是一个anchor的[width, height, x_center, y_center]
    (input :anchor--[xmin, ymin, xmax, ymax]
    output:width, height, x_center, y_center of  the anchor)
    '''
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    '''
    输入的是一系列矩阵，它们weight，height不同，center_x，center_y相同
    输出的是一系列矩阵，格式为[[xmin, ymin, xmax, ymax],[...]...]
    (input：a set of anchors which have different weight 、height
            and have the same center_x 、center_y)
      output: a set if anchors which format is [[xmin, ymin, xmax, ymax],[...]...]
    )
    '''
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    '''
    输入：参考anchor（16*16）与缩放比集合ratios
    输出：三个anchor，它们的面积分为参考anchor面积的0.5，1，2倍
    (input:reference anchor(16*16)、ratios
     output:a set of three anchors.Their areas are 0.5,1,2 times of the reference anchor )
    '''
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratio = size / ratios
    ws = np.round(np.sqrt(size_ratio))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

#generate_anchors()
