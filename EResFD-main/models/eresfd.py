"""
EResFD
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from layers.functions.prior_box import _get_prior_box
from layers.functions.detection import Detect
from data.config import cfg
from models.eresfd_16 import (
    _get_base_layers,
    _get_head_layers,
    _forward_base,
    _forward_head,
)
from typing import List, Tuple

def build_model(phase, num_classes=2, width_mult=1.0, weights_init=None):
    """ Builds and returns the EResFD model """
    base_ = _get_base_layers(width_mult=width_mult)

    # if multiple anchor scales per pixel location
    if any(
        isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
        for anchors_per_layer in cfg.ANCHOR_SIZES
    ):
        # if multiple anchor size ratio per pixel location
        if any(
            isinstance(anchor_size_ratio_per_layer, list)
            and len(anchor_size_ratio_per_layer) > 1
            for anchor_size_ratio_per_layer in cfg.ANCHOR_SIZE_RATIO
        ):
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_sizes=cfg.ANCHOR_SIZES,
                anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
            )
        else:
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_sizes=cfg.ANCHOR_SIZES,
            )
    else:
        if any(
            isinstance(anchor_size_ratio_per_layer, list)
            and len(anchor_size_ratio_per_layer) > 1
            for anchor_size_ratio_per_layer in cfg.ANCHOR_SIZE_RATIO
        ):
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
            )
        else:
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
            )

    return EResFD(
        phase,
        base_,
        head_,
        num_classes,
        _forward_base,
        _forward_head,
        anchor_steps=cfg.STEPS,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_clip=cfg.CLIP,
        anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
        weights_init=weights_init,
    )

def build_model_for_torchscript(phase, num_classes=2, width_mult=1.0, weights_init=None):
    """ Builds and returns the EResFD model to be used for torchscript """
    base_ = _get_base_layers(width_mult=width_mult)

    # if multiple anchor scales per pixel location
    if any(
        isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
        for anchors_per_layer in cfg.ANCHOR_SIZES
    ):
        # if multiple anchor size ratio per pixel location
        if any(
            isinstance(anchor_size_ratio_per_layer, list)
            and len(anchor_size_ratio_per_layer) > 1
            for anchor_size_ratio_per_layer in cfg.ANCHOR_SIZE_RATIO
        ):
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_sizes=cfg.ANCHOR_SIZES,
                anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
            )
        else:
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_sizes=cfg.ANCHOR_SIZES,
            )
    else:
        if any(
            isinstance(anchor_size_ratio_per_layer, list)
            and len(anchor_size_ratio_per_layer) > 1
            for anchor_size_ratio_per_layer in cfg.ANCHOR_SIZE_RATIO
        ):
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
                anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
            )
        else:
            head_ = _get_head_layers(
                width_mult=width_mult,
                num_classes=num_classes,
                maxout_bg_size=cfg.MAXOUT_BG_SIZE,
            )

    return EResFDTorchscript(
        phase,
        base_,
        head_,
        num_classes,
        _forward_base,
        _forward_head,
        anchor_steps=cfg.STEPS,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_clip=cfg.CLIP,
        anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
        weights_init=weights_init,
    )


class EResFD(nn.Module):
    def __init__(
        self,
        phase,
        base,
        head: List[List[torch.nn.Module]],
        num_classes,
        _forward_base,
        _forward_head,
        anchor_steps=cfg.STEPS,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_clip=cfg.CLIP,
        anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
        weights_init=None,
    ):
        super(EResFD, self).__init__()

        self.phase = phase
        self.num_classes = num_classes

        # base
        self.base = base

        # head
        self.act_loc = nn.ModuleList(head[0])
        self.act_conf = nn.ModuleList(head[1])
        self.loc = nn.ModuleList(head[2])
        self.conf = nn.ModuleList(head[3])

        self.loc0 = head[2][0]
        self.loc1 = head[2][1]
        self.loc2 = head[2][2]
        self.loc3 = head[2][3]
        self.loc4 = head[2][4]
        self.loc5 = head[2][5]

        self.conf0 = head[3][0]
        self.conf1 = head[3][1]
        self.conf2 = head[3][2]
        self.conf3 = head[3][3]
        self.conf4 = head[3][4]
        self.conf5 = head[3][5]

        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

        self._forward_base = _forward_base

        self.anchor_steps = anchor_steps
        self.anchor_sizes = anchor_sizes
        self.anchor_size_ratio = anchor_size_ratio
        self.anchor_clip = anchor_clip

        if any(
            isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
            for anchors_per_layer in self.anchor_sizes
        ):
            self.multiple_anchorscales_per_layer = True
        else:
            self.multiple_anchorscales_per_layer = False

        if any(
            isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
            for anchors_per_layer in self.anchor_size_ratio
        ):
            self.multiple_anchorsratios_per_layer = True
        else:
            self.multiple_anchorsratios_per_layer = False

        self.weights_init_type = weights_init
        # self.feature_maps_sizes: List[List[List[int]]] = []
        self.priors = torch.empty(size=(0,))

    def forward(self, x):
        # NOTE:
        # sources: a list of six [b, c, h, w] tensors
        sources = self.base(x)

        # NOTE:
        # conf: a list of siz [b, h, w, num_class] tensors
        # loc:  a list of siz [b, h, w, 4] tensors
        # if multiple anchor scales per pixel location
        if self.multiple_anchorscales_per_layer:
            if self.multiple_anchorsratios_per_layer:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc,
                    # self.act_conf,
                    # self.loc,
                    # self.conf,
                    sources,
                    anchor_sizes=self.anchor_sizes,
                    anchor_size_ratio=self.anchor_size_ratio,
                )
            else:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc,
                    # self.act_conf,
                    # self.loc,
                    # self.conf,
                    sources,
                    anchor_sizes=self.anchor_sizes,
                )
        else:
            if self.multiple_anchorsratios_per_layer:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc,
                    # self.act_conf,
                    # self.loc,
                    # self.conf,
                    sources,
                    anchor_size_ratio=self.anchor_size_ratio,
                )
            else:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc, self.act_conf, self.loc, self.conf,
                    sources
                )

        # NOTE: can be precomputed
        # noting but just store spatial resolution of each feature-map
        # for example, resolution of an input is 256 by 256
        # features_maps is to be [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2]]
        features_maps: List[List[int]] = []
        # for i in range(len(loc_over_stack[-1])):
        #     feat = []
        #     feat += [loc_over_stack[-1][i].size(1), loc_over_stack[-1][i].size(2)]
        #     features_maps += [feat]

        for i in range(len(loc_over_stack[-1])):
            # feat: List[List[int]] = []
            temp_list = [loc_over_stack[-1][i].size(1), loc_over_stack[-1][i].size(2)]
            # feat.append(temp_list)
            features_maps += [temp_list]

        # self.feature_maps_sizes = features_maps

        # NOTE: can be precomputed
        original_size: List[int] = list(x.size()[2:])
        with torch.no_grad():
            # cfg.STEPS: [4, 8, 16, 32, 64, 128]
            # cfg.ANCHOR_SIZES: [16, 32, 64, 128, 256, 512]
            # cfg.CLIP: False
            self.priors = _get_prior_box(
                original_size,
                self.anchor_steps,
                self.anchor_sizes,
                self.anchor_clip,
                features_maps,
            )

        # concat. all features for all scale
        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for loc, conf in zip(loc_over_stack, conf_over_stack):
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            outputs.append((loc, conf))

        if self.phase == "test":
            loc = outputs[-1][0]
            conf = outputs[-1][1]
            self.priors = self.priors.cuda()
            output = self.detect(
                # loc preds (1, # anchors, 4)
                loc.view(loc.size(0), -1, 4),
                # conf preds (1, # anchors, 2)
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                # default box (# boxes, 4)
                # 4: [center_x, center_y, scaled_h, scaled_w]
                self.priors, #.type(type(x.data)), # changed for torchscript
            )
        else:
            outputs_with_priors = []
            for out in outputs:
                loc = out[0]
                conf = out[1]

                loc = loc.view(loc.size(0), -1, 4)
                conf = conf.view(conf.size(0), -1, self.num_classes)

                output = (loc, conf, self.priors)

                outputs_with_priors.append(output)

            output = outputs_with_priors

        return output

    def custom_forward_head(
            self,
            # act_loc_layers: torch.nn.ModuleList,
            # act_conf_layers: torch.nn.ModuleList,
            # loc_layers: torch.nn.ModuleList,
            # conf_layers: torch.nn.ModuleList,
            input: List[List[torch.Tensor]],
            maxout_bg_size: int = 3,
            anchor_sizes: List[int] = (),
            anchor_size_ratio: List[float] = (),
            ):
        """ Head forward function, custom-made so that it works with Torchscript """
        loc_outputs_over_stack: List[List[torch.Tensor]] = list()
        conf_outputs_over_stack: List[List[torch.Tensor]] = list()

        for stack_id in range(len(input)):
            num_anchor_per_pixel = 1
            if len(anchor_sizes) :
                num_anchor_per_pixel *= len(anchor_sizes[0])
            if len(anchor_size_ratio):
                num_anchor_per_pixel *= len(anchor_size_ratio[0])

            loc_outputs = list()
            conf_outputs = list()

            x = input[stack_id]

            # loc_x = act_loc_layers[0](x[0])
            loc_x = torch.nn.functional.relu(x[0])
            # conf_x = act_conf_layers[0](x[0])
            conf_x = torch.nn.functional.relu(x[0])

            loc_x = self.loc0(loc_x)
            conf_x = self.conf0(conf_x)

            # for dealing with multiple anchors on each pixel
            if len(anchor_sizes) or len(anchor_size_ratio) :
                conf_x_per_pixel = []
                for i in range(num_anchor_per_pixel):
                    start_idx = 4 * i
                    # for dealing with maxout BG labels
                    maxout_bg_conf, _ = torch.max(
                        conf_x[:, start_idx: start_idx + maxout_bg_size, :, :],
                        dim=1,
                        keepdim=True,
                    )
                    fg_conf = conf_x[
                              :, start_idx + maxout_bg_size: start_idx + maxout_bg_size + 1, :, :
                              ]
                    conf_x_per_pixel += [maxout_bg_conf, fg_conf]

                conf_x = torch.cat(conf_x_per_pixel, dim=1)
            else:
                max_conf, _ = torch.max(
                    conf_x[:, 0:maxout_bg_size, :, :], dim=1, keepdim=True
                )
                conf_x = torch.cat(
                    (max_conf, conf_x[:, maxout_bg_size: maxout_bg_size + 1, :, :]), dim=1
                )

            loc_outputs.append(loc_x.permute(0, 2, 3, 1).contiguous())
            conf_outputs.append(conf_x.permute(0, 2, 3, 1).contiguous())

            # assert len(x) == 6
            # for i in range(1, len(x)):
            #     conf_outputs.append(
            #         conf_layers[i](act_conf_layers[i](x[i]))
            #         .permute(0, 2, 3, 1)
            #         .contiguous()
            #     )
            #     loc_outputs.append(
            #         loc_layers[i](act_loc_layers[i](x[i])).permute(0, 2, 3, 1).contiguous()
            #     )
            # conf_outputs.append(
            #             self.conf0(torch.nn.functional.relu((x[0])))
            #             .permute(0, 2, 3, 1)
            #             .contiguous()
            #         )
            # loc_outputs.append(
            #             self.loc0(torch.nn.functional.relu((x[0]))).permute(0, 2, 3, 1).contiguous()
            #         )
            conf_outputs.append(
                self.conf1(torch.nn.functional.relu((x[1])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc1(torch.nn.functional.relu((x[1]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf2(torch.nn.functional.relu((x[2])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc2(torch.nn.functional.relu((x[2]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf3(torch.nn.functional.relu((x[3])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc3(torch.nn.functional.relu((x[3]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf4(torch.nn.functional.relu((x[4])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc4(torch.nn.functional.relu((x[4]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf5(torch.nn.functional.relu((x[5])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc5(torch.nn.functional.relu((x[5]))).permute(0, 2, 3, 1).contiguous()
            )

            conf_outputs_over_stack.append(conf_outputs)
            loc_outputs_over_stack.append(loc_outputs)

        return conf_outputs_over_stack, loc_outputs_over_stack

    def load_weights(self, filename):
        base = torch.load(
            filename + ".base.pth", map_location=lambda storage, loc: storage
        )
        self.base.load_state_dict(base)
        act_loc = torch.load(
            filename + ".act_loc.pth", map_location=lambda storage, loc: storage
        )
        self.act_loc.load_state_dict(act_loc)
        act_conf = torch.load(
            filename + ".act_conf.pth", map_location=lambda storage, loc: storage
        )
        self.act_conf.load_state_dict(act_conf)
        loc = torch.load(
            filename + ".loc.pth", map_location=lambda storage, loc: storage
        )
        self.loc.load_state_dict(loc)
        conf = torch.load(
            filename + ".conf.pth", map_location=lambda storage, loc: storage
        )
        self.conf.load_state_dict(conf)
        return True

    def save_weights(self, filename):
        torch.save(self.base.cpu().state_dict(), filename + ".base.pth")
        torch.save(self.act_loc.cpu().state_dict(), filename + ".act_loc.pth")
        torch.save(self.act_conf.cpu().state_dict(), filename + ".act_conf.pth")
        torch.save(self.loc.cpu().state_dict(), filename + ".loc.pth")
        torch.save(self.conf.cpu().state_dict(), filename + ".conf.pth")
        return True

    def xavier(self, param):
        init.xavier_uniform_(param)

    def kaiming(self, param):
        init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def normal(self, param, std=0.01):
        init.normal_(param, mean=0.0, std=std)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            if self.weights_init_type == "xavier":
                self.xavier(m.weight.data)
            elif self.weights_init_type == "kaiming":
                self.kaiming(m.weight.data)
            else:
                self.normal(m.weight.data, std=0.01)
            if m.bias is not None:
                m.bias.data.zero_()

class EResFDTorchscript(nn.Module):
    """ EResFD model, but can me converted to torchscript for mobile development """
    def __init__(
        self,
        phase,
        base,
        head: List[List[torch.nn.Module]],
        num_classes,
        _forward_base,
        _forward_head,
        anchor_steps=cfg.STEPS,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_clip=cfg.CLIP,
        anchor_size_ratio=cfg.ANCHOR_SIZE_RATIO,
        weights_init=None,
    ):
        super(EResFDTorchscript, self).__init__()

        self.phase = phase
        self.num_classes = num_classes

        # base
        self.base = base

        # head
        self.act_loc = nn.ModuleList(head[0])
        self.act_conf = nn.ModuleList(head[1])
        self.loc = nn.ModuleList(head[2])
        self.conf = nn.ModuleList(head[3])

        self.loc0 = head[2][0]
        self.loc1 = head[2][1]
        self.loc2 = head[2][2]
        self.loc3 = head[2][3]
        self.loc4 = head[2][4]
        self.loc5 = head[2][5]

        self.conf0 = head[3][0]
        self.conf1 = head[3][1]
        self.conf2 = head[3][2]
        self.conf3 = head[3][3]
        self.conf4 = head[3][4]
        self.conf5 = head[3][5]

        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

        self._forward_base = _forward_base

        self.anchor_steps = anchor_steps
        self.anchor_sizes = anchor_sizes
        self.anchor_size_ratio = anchor_size_ratio
        self.anchor_clip = anchor_clip

        if any(
            isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
            for anchors_per_layer in self.anchor_sizes
        ):
            self.multiple_anchorscales_per_layer = True
        else:
            self.multiple_anchorscales_per_layer = False

        if any(
            isinstance(anchors_per_layer, list) and len(anchors_per_layer) > 1
            for anchors_per_layer in self.anchor_size_ratio
        ):
            self.multiple_anchorsratios_per_layer = True
        else:
            self.multiple_anchorsratios_per_layer = False

        self.weights_init_type = weights_init
        # self.feature_maps_sizes: List[List[List[int]]] = []
        self.priors = torch.empty(size=(0,))

    def forward(self, x):
        # Preprocessing
        x = torch.permute(x, (2, 0, 1))
        # RBG to BGR
        x = x[[2, 1, 0], :, :].type(torch.float32)

        img_mean = torch.tensor([104, 117, 123],dtype=torch.float32).view(3, 1, 1)
        x -= img_mean
        x = x[[2, 1, 0], :, :]

        x = x[None, :, :, :]
        # NOTE:
        # sources: a list of six [b, c, h, w] tensors
        sources = self.base(x)

        # NOTE:
        # conf: a list of siz [b, h, w, num_class] tensors
        # loc:  a list of siz [b, h, w, 4] tensors
        # if multiple anchor scales per pixel location
        if self.multiple_anchorscales_per_layer:
            if self.multiple_anchorsratios_per_layer:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc,
                    # self.act_conf,
                    # self.loc,
                    # self.conf,
                    sources,
                    anchor_sizes=self.anchor_sizes,
                    anchor_size_ratio=self.anchor_size_ratio,
                )
            else:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc,
                    # self.act_conf,
                    # self.loc,
                    # self.conf,
                    sources,
                    anchor_sizes=self.anchor_sizes,
                )
        else:
            if self.multiple_anchorsratios_per_layer:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc,
                    # self.act_conf,
                    # self.loc,
                    # self.conf,
                    sources,
                    anchor_size_ratio=self.anchor_size_ratio,
                )
            else:
                conf_over_stack, loc_over_stack = self.custom_forward_head(
                    # self.act_loc, self.act_conf, self.loc, self.conf,
                    sources
                )

        # NOTE: can be precomputed
        # noting but just store spatial resolution of each feature-map
        # for example, resolution of an input is 256 by 256
        # features_maps is to be [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2]]
        features_maps: List[List[int]] = []
        # for i in range(len(loc_over_stack[-1])):
        #     feat = []
        #     feat += [loc_over_stack[-1][i].size(1), loc_over_stack[-1][i].size(2)]
        #     features_maps += [feat]

        for i in range(len(loc_over_stack[-1])):
            # feat: List[List[int]] = []
            temp_list = [loc_over_stack[-1][i].size(1), loc_over_stack[-1][i].size(2)]
            # feat.append(temp_list)
            features_maps += [temp_list]

        # self.feature_maps_sizes = features_maps

        # NOTE: can be precomputed
        original_size: List[int] = list(x.size()[2:])
        # cfg.STEPS: [4, 8, 16, 32, 64, 128]
        # cfg.ANCHOR_SIZES: [16, 32, 64, 128, 256, 512]
        # cfg.CLIP: False
        self.priors = _get_prior_box(
            original_size,
            self.anchor_steps,
            self.anchor_sizes,
            self.anchor_clip,
            features_maps,
        )

        # concat. all features for all scale
        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for loc, conf in zip(loc_over_stack, conf_over_stack):
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            outputs.append((loc, conf))

        if self.phase == "test":
            loc = outputs[-1][0]
            conf = outputs[-1][1]
            self.priors = self.priors
            output = self.detect(
                # loc preds (1, # anchors, 4)
                loc.view(loc.size(0), -1, 4),
                # conf preds (1, # anchors, 2)
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                # default box (# boxes, 4)
                # 4: [center_x, center_y, scaled_h, scaled_w]
                self.priors, #.type(type(x.data)), # changed for torchscript
            )
        else:
            raise ValueError('training, this class is only meant for testing!')

        return output

    def custom_forward_head(
            self,
            # act_loc_layers: torch.nn.ModuleList,
            # act_conf_layers: torch.nn.ModuleList,
            # loc_layers: torch.nn.ModuleList,
            # conf_layers: torch.nn.ModuleList,
            input: List[List[torch.Tensor]],
            maxout_bg_size: int = 3,
            anchor_sizes: List[int] = (),
            anchor_size_ratio: List[float] = (),
            ):
        """ Head forward function, custom-made so that it works with Torchscript """
        loc_outputs_over_stack: List[List[torch.Tensor]] = list()
        conf_outputs_over_stack: List[List[torch.Tensor]] = list()

        for stack_id in range(len(input)):
            num_anchor_per_pixel = 1
            if len(anchor_sizes) :
                num_anchor_per_pixel *= len(anchor_sizes[0])
            if len(anchor_size_ratio):
                num_anchor_per_pixel *= len(anchor_size_ratio[0])

            loc_outputs = list()
            conf_outputs = list()

            x = input[stack_id]

            # loc_x = act_loc_layers[0](x[0])
            loc_x = torch.nn.functional.relu(x[0])
            # conf_x = act_conf_layers[0](x[0])
            conf_x = torch.nn.functional.relu(x[0])

            loc_x = self.loc0(loc_x)
            conf_x = self.conf0(conf_x)

            # for dealing with multiple anchors on each pixel
            if len(anchor_sizes) or len(anchor_size_ratio) :
                conf_x_per_pixel = []
                for i in range(num_anchor_per_pixel):
                    start_idx = 4 * i
                    # for dealing with maxout BG labels
                    maxout_bg_conf, _ = torch.max(
                        conf_x[:, start_idx: start_idx + maxout_bg_size, :, :],
                        dim=1,
                        keepdim=True,
                    )
                    fg_conf = conf_x[
                              :, start_idx + maxout_bg_size: start_idx + maxout_bg_size + 1, :, :
                              ]
                    conf_x_per_pixel += [maxout_bg_conf, fg_conf]

                conf_x = torch.cat(conf_x_per_pixel, dim=1)
            else:
                max_conf, _ = torch.max(
                    conf_x[:, 0:maxout_bg_size, :, :], dim=1, keepdim=True
                )
                conf_x = torch.cat(
                    (max_conf, conf_x[:, maxout_bg_size: maxout_bg_size + 1, :, :]), dim=1
                )

            loc_outputs.append(loc_x.permute(0, 2, 3, 1).contiguous())
            conf_outputs.append(conf_x.permute(0, 2, 3, 1).contiguous())

            # assert len(x) == 6
            # for i in range(1, len(x)):
            #     conf_outputs.append(
            #         conf_layers[i](act_conf_layers[i](x[i]))
            #         .permute(0, 2, 3, 1)
            #         .contiguous()
            #     )
            #     loc_outputs.append(
            #         loc_layers[i](act_loc_layers[i](x[i])).permute(0, 2, 3, 1).contiguous()
            #     )
            # conf_outputs.append(
            #             self.conf0(torch.nn.functional.relu((x[0])))
            #             .permute(0, 2, 3, 1)
            #             .contiguous()
            #         )
            # loc_outputs.append(
            #             self.loc0(torch.nn.functional.relu((x[0]))).permute(0, 2, 3, 1).contiguous()
            #         )
            conf_outputs.append(
                self.conf1(torch.nn.functional.relu((x[1])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc1(torch.nn.functional.relu((x[1]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf2(torch.nn.functional.relu((x[2])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc2(torch.nn.functional.relu((x[2]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf3(torch.nn.functional.relu((x[3])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc3(torch.nn.functional.relu((x[3]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf4(torch.nn.functional.relu((x[4])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc4(torch.nn.functional.relu((x[4]))).permute(0, 2, 3, 1).contiguous()
            )
            conf_outputs.append(
                self.conf5(torch.nn.functional.relu((x[5])))
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            loc_outputs.append(
                self.loc5(torch.nn.functional.relu((x[5]))).permute(0, 2, 3, 1).contiguous()
            )

            conf_outputs_over_stack.append(conf_outputs)
            loc_outputs_over_stack.append(loc_outputs)

        return conf_outputs_over_stack, loc_outputs_over_stack

    def load_weights(self, filename):
        base = torch.load(
            filename + ".base.pth", map_location=lambda storage, loc: storage
        )
        self.base.load_state_dict(base)
        act_loc = torch.load(
            filename + ".act_loc.pth", map_location=lambda storage, loc: storage
        )
        self.act_loc.load_state_dict(act_loc)
        act_conf = torch.load(
            filename + ".act_conf.pth", map_location=lambda storage, loc: storage
        )
        self.act_conf.load_state_dict(act_conf)
        loc = torch.load(
            filename + ".loc.pth", map_location=lambda storage, loc: storage
        )
        self.loc.load_state_dict(loc)
        conf = torch.load(
            filename + ".conf.pth", map_location=lambda storage, loc: storage
        )
        self.conf.load_state_dict(conf)
        return True

    def save_weights(self, filename):
        torch.save(self.base.cpu().state_dict(), filename + ".base.pth")
        torch.save(self.act_loc.cpu().state_dict(), filename + ".act_loc.pth")
        torch.save(self.act_conf.cpu().state_dict(), filename + ".act_conf.pth")
        torch.save(self.loc.cpu().state_dict(), filename + ".loc.pth")
        torch.save(self.conf.cpu().state_dict(), filename + ".conf.pth")
        return True

    def xavier(self, param):
        init.xavier_uniform_(param)

    def kaiming(self, param):
        init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def normal(self, param, std=0.01):
        init.normal_(param, mean=0.0, std=std)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            if self.weights_init_type == "xavier":
                self.xavier(m.weight.data)
            elif self.weights_init_type == "kaiming":
                self.kaiming(m.weight.data)
            else:
                self.normal(m.weight.data, std=0.01)
            if m.bias is not None:
                m.bias.data.zero_()


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image

if __name__ == "__main__":
    cfg.ANCHOR_SIZE_RATIO = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    net = build_model("train", num_classes=2, width_mult=0.0625)
    net.load_weights("./weights/eresnet_sepfpn_cpm_reproduce.pth")
    inputs = Variable(torch.randn(4, 3, 640, 640))
    output = net(inputs)
