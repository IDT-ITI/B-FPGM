"""
Code Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/prior_box.py
"""
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
from itertools import product as product
import math
from data.config import cfg as globalcfg
from typing import List

def _get_prior_box(original_size: List[int], steps: List[int], min_sizes: List[int], clip:bool, feature_maps: List[List[int]]) -> torch.Tensor:
    # original_size: size of input
    # steps: stride from the size of input
    # min_sizes: ancor-size (scale)
    # clip: False
    # feature_maps: size of featuremaps

    anchor_size_ratio = [1.0]
    imh, imw = original_size
    mean: List[float] = []

    for k in range(len(feature_maps)):

        feath = feature_maps[k][0]
        featw = feature_maps[k][1]

        f_kw = imw / steps[k]
        f_kh = imh / steps[k]

        # NOTE debugging purpose
        # output_each_scale = []
        list1: List[int] = [_ for _ in range(feath)]
        list2: List[int] = [_ for _ in range(featw)]
        for i in list1:
            for j in list2:
                # get center point which is normalized by the feature-map size
                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                # get anchor-size normalized by original input-size

                # If anchor size settings are [16, 32, 64, 128, [256, 512]]
                # i.e. If one layer wants to have various anchor sizes,
                # NOTE: The sequence order of priors corresponding to the anchor scale (256, 512) doesn't have to be ordered.
                # NOTE: Because we didn't determinstically set the output of convolution to regress and classify specific anchor size on specific channel.
                # NOTE: It only matters if these anchor boxes are inside the layer boundary.
                # NOTE: e.g. in feature maps size setting [160, 80, 40, 20 , 10],
                # NOTE: anchor size setting [16, 32 , 64, 128 , p256]
                #
                # if type(min_sizes[k]) == list: # Commented for torchscript
                #     for anchor_size in min_sizes[k]:
                #         s_kw = anchor_size / imw
                #         s_kh = anchor_size / imh
                #
                #         # diverse anchor ratios
                #         if type(anchor_size_ratio) == list:
                #             if type(anchor_size_ratio[0]) == list:
                #                 # if 2d-list (ratio is different for each detection layers)
                #                 for ratio in anchor_size_ratio[k]:
                #                     mean += [cx, cy, s_kw, s_kh * ratio]
                #             else:
                #                 # if 1d-list (ratio is same for all detection layers)
                #                 for ratio in anchor_size_ratio:
                #                     mean += [cx, cy, s_kw, s_kh * ratio]
                #         else:
                #             mean += [cx, cy, s_kw, s_kh * anchor_size_ratio]
                # else:
                s_kw = min_sizes[k] / imw
                s_kh = min_sizes[k] / imh

                # diverse anchor ratios
                # if type(anchor_size_ratio) == list:
                #     if type(anchor_size_ratio[0]) == list:
                #         # if 2d-list (ratio is different for each detection layers)
                #         for ratio in anchor_size_ratio[k]:
                #             mean += [cx, cy, s_kw, s_kh * ratio]
                #     else:
                #         # if 1d-list (ratio is same for all detection layers)
                #         for ratio in anchor_size_ratio:
                #             mean += [cx, cy, s_kw, s_kh * ratio]
                # else:
                #     mean += [cx, cy, s_kw, s_kh * anchor_size_ratio]

                # Commented for torchscript
                for ratio in anchor_size_ratio:
                    mean += [cx, cy, s_kw, s_kh * ratio]

    output = torch.tensor(mean).view(-1, 4)

    if clip:
        output.clamp_(max=1, min=0)

    return output
