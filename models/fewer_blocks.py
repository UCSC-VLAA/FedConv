
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
    
class fedconv_block_nonnorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                act_layer=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d, drop_path=None, kernel_size=3,
                layer_scale_init_value=0., drop_block=None,):
        super(fedconv_block_nonnorm, self).__init__()

        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        pad = get_padding(kernel_size, stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=pad,  groups=planes)  #depthwise conv

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1)

        self.act3 = act_layer
        self.downsample = downsample
        self.drop_path = drop_path
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(outplanes)) if layer_scale_init_value > 0 else None


    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x
    

class fedconv_invert_block_nonnorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                act_layer=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d, drop_path=None, kernel_size=3,
                layer_scale_init_value=0.,drop_block=None,):
        super(fedconv_invert_block_nonnorm, self).__init__()

        innerplanes = planes * self.expansion
        pad = get_padding(kernel_size, stride)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1)
        self.act1 = act_layer
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel_size, stride=stride, padding=pad,  groups=innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, planes, kernel_size=1)

        self.downsample = downsample
        self.drop_path = drop_path
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(planes)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
            
        x += shortcut

        return x



class fedconv_invertup_block_nonnorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                act_layer=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d, drop_path=None, kernel_size=3,
                layer_scale_init_value=0. ,drop_block=None,
                ):
        super(fedconv_invertup_block_nonnorm, self).__init__()

        innerplanes = planes * self.expansion
        pad = get_padding(kernel_size, stride)

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride, padding=pad,  groups=inplanes)

        self.conv2 = nn.Conv2d(inplanes, innerplanes, kernel_size=1)
        self.act2 = act_layer

        self.conv3 = nn.Conv2d(innerplanes, planes, kernel_size=1)

        self.downsample = downsample
        self.drop_path = drop_path
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(planes)) if layer_scale_init_value > 0 else None


    def forward(self, x):
        shortcut = x

        x = self.conv1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut

        return x