# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
Aug 22, 2020
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from typing import TypeVar, Union, Optional, Tuple
import math
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



blocks_dict = {
    'BASIC': BasicBlock,
}

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)
        
class SqueezeExcite(nn.Module):

    def __init__(self, channels: int, factor: Union[int, float] = 1/4):
        super().__init__()
        self.squeeze_excite = nn.Sequential(
            nn.Linear(channels, math.floor(channels * factor)),
            nn.ReLU(),
            nn.Linear(math.floor(channels * factor), channels),
            nn.Hardsigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.size()
        x = F.adaptive_max_pool2d(x, 1).view(b, c)
        return self.squeeze_excite(x).view(b, c, 1, 1)

class ConvBN(nn.Module):

    def __init__(
            self,
            in_c: int,
            out_c: int,
            k_size: int,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            activation : Optional[nn.Module] = None,
            squeeze_excite: bool = False
    ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k_size = k_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.padding = int((k_size - 1) * dilation / 2)       # same padding
        self.squeeze_excite = squeeze_excite

        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, stride, self.padding,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM)
        )
        if activation:
            self.conv_bn.add_module("2", activation)

        if self.squeeze_excite:
            self.squeeze = SqueezeExcite(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_bn(x)
        if self.squeeze_excite:
            x = x * self.squeeze(x)
        return x

class ASPP(nn.Module):

    def __init__(self, shallow_channels: int, deep_channels: int, n_class: int,
                 head_c: int = 32, dilations: list = [1, 3, 5]):
        super().__init__()
        self.shallow_conv = nn.Conv2d(shallow_channels, n_class, 1)

        self.pyramid = nn.ModuleList()
        self.pyramid.append(nn.Conv2d(deep_channels, head_c, 1))
        kernel_size = 3
        for dilation in dilations:
            padding = int((kernel_size - 1) * dilation / 2)
            self.pyramid.append(nn.Conv2d(deep_channels, head_c, kernel_size,
                                          dilation=dilation, padding=padding))

        self.deep_conv = nn.Conv2d(head_c * len(self.pyramid) + deep_channels,
                                   n_class, 1)

    def forward(self, shallow_x, deep_x):
        # prepare skip connection from low-level features
        shallow_x = self.shallow_conv(shallow_x)

        # prepare high-level features
        # (skip-connection, 1x1-conv, atrous spatial pyramid pooling)
        deep_x_cat = [deep_x] + [module(deep_x)for module in self.pyramid]
        deep_x_cat = torch.cat(deep_x_cat, dim=1)
        deep_x = self.deep_conv(deep_x_cat)
        deep_x = F.interpolate(deep_x, size=shallow_x.shape[-2:],
                                       mode="bilinear", align_corners=True)

        # add skip connection from low-level features
        segmentation_x = deep_x + shallow_x
        segmentation_x = F.interpolate(segmentation_x, scale_factor=2,
                                       mode="bilinear", align_corners=True)

        return segmentation_x


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DCT_Stream(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(DCT_Stream, self).__init__()

        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(in_channels=21,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      dilation=8,
                      padding=8),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.dc_layer2 = self._make_layer(BasicBlock, inplanes=6, planes=3, blocks=4, stride=1)

        last_inp_channels = 9
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        # mobilenet
        input_channel = 16
        last_channel = 1280
        mode='small'
        width_mult=1.0

        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            layer1_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2]             
            ]
            layer2_setting = [
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1], 
                ]
            layer3_setting = [
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1]
            ]
            layer4_setting = [
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1]
                ]
            layer5_setting = [
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1], 
                ]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features1 = conv_bn(3, input_channel, 2, nlin_layer=Hswish)
        self.classifier = []

        # building mobile blocks
        self.layer1, input_channel = self._make_layer_bottleneck(input_channel, layer1_setting, width_mult)
        self.layer2, input_channel = self._make_layer_bottleneck(input_channel, layer2_setting, width_mult)
        self.layer3, input_channel = self._make_layer_bottleneck(input_channel, layer3_setting, width_mult)
        self.layer4, input_channel = self._make_layer_bottleneck(input_channel, layer4_setting, width_mult)
        self.layer5, input_channel = self._make_layer_bottleneck(input_channel, layer5_setting, width_mult)

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
            self.features = nn.Sequential(*self.features)
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features2 = conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.8),    # refer to paper section 6
            nn.Linear(last_channel, 2),
        )
        self.head = ASPP(shallow_channels=24, deep_channels=48, n_class=2, head_c=128)

    def _make_layer_bottleneck(self, input_channel, block_setting, width_mult):
        for k, exp, c, se, nl, s in block_setting:
            layers = list()
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            layers.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
            return nn.Sequential(*layers), input_channel

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, qtable):
        # B, 21, 512, 512
        x = self.dc_layer0_dil(x) # B, 64, 512, 512
        x1 = self.dc_layer1_tail(x) # B, 3, 512, 512
        B, C, H, W = x1.shape
        
        x_temp = x1.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4) # [B, C, 8, 8, 64, 64]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1) # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, 3, 8, 8, 64, 64]
        x2 = xq_temp.reshape(B, C, H, W) # [B, 3, 512, 512]
        x = torch.cat([x1, x2], dim=1) # [B, 6, 512, 512]
        x = self.dc_layer2(x)  # # [B, 3, 512, 512]
        # entrada mobilenetv3
        early = self.features1(x) # [B, 16, 256, 256]
        print(early.is_cuda)
        c1 = self.layer1(early) # [B, 16, 128, 128] 1/4
        c2 = self.layer2(c1) # [B, 24, 64, 64] 1/8
        c3 = self.layer3(c2) # [B, 40, 32, 32] 1/16
        c4 = self.layer4(c3) # [B, 48, 32, 32] 1/16
        c5 = self.layer5(c4) # [B, 96, 16, 16]
        cout = self.features2(c5) # [B, 576, 16, 16]
        #segmentación ASPP
        x = self.head(c2, c4) # [B, 2, 128, 128]
        print(x.is_cuda)
        return x

    def init_weights(self, pretrained_rgb='', pretrained_dct='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained_dct):
            loaded_dict = torch.load(pretrained_dct)['state_dict']
            model_dict = self.state_dict()
            loaded_dict = {k: v for k, v in loaded_dict.items()
                               if k in model_dict.keys()}
            loaded_dict = {k:v for k,v in loaded_dict.items()
                           if not k.startswith('last_layer')}
            logger.info('=> (DCT) loading pretrained model {} ({})'.format(pretrained_dct, len(loaded_dict)))
            model_dict.update(loaded_dict)
            self.load_state_dict(model_dict)
        else:
            logger.warning('=> Cannot load pretrained DCT')


def get_mobileNet_seg(cfg, **kwargs):
    model = DCT_Stream(cfg, **kwargs)
    # print(model)
    model.init_weights(cfg.MODEL.PRETRAINED_RGB, cfg.MODEL.PRETRAINED_DCT)

    return model
