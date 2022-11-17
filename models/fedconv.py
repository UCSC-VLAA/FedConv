import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, SelectAdaptivePool2d,create_classifier,trunc_normal_
from .registry import register_model
from .fx_features import register_notrace_module
from collections import OrderedDict
from .fewer_blocks import *
from torch import Tensor

__all__ = ['fedconv_base','fedconv_invert','fedconv_invertup']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'fedconv':_cfg(interpolation='bicubic'),
}


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _norm(norm_layer, num_features, group_num=2):
    if norm_layer == nn.GroupNorm:
        return norm_layer(group_num, num_features)  #default: 2 group, between real ln and instance norm
    elif norm_layer==nn.LayerNorm:
        return nn.GroupNorm(1, num_features)  # same as real LN
    else:
        return norm_layer(num_features)

class UnderSp(nn.Module):
    '''
    function of softplus -value
    '''
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20, value: float = 1.) -> None:
        super(UnderSp, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)- self.value

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    #p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
        _norm(norm_layer, out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        _norm(norm_layer, out_channels)
    ])


def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., stride_stage=1,
        invert=False, **kwargs):
    stages = []

    net_num_blocks = sum(block_repeats)
    net_block_idx = 0

    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx < stride_stage else 2

        ds_out = planes * block_fn.expansion if not invert else planes
        if stride != 1 or inplanes != ds_out:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=ds_out, kernel_size=down_kernel_size,
                stride=stride, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)
        else:
            downsample = None

        block_kwargs = dict(drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule

            blocks.append(block_fn(
                inplanes, planes, stride, downsample,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            
            inplanes = planes * block_fn.expansion  if not invert else planes
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))

    return stages


class Fedconv(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 output_stride=32, down_kernel_size=1, avg_down=False,
                 act_layer=nn.GELU(), norm_layer=LayerNorm2d, 
                 drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', block_args=None,
                 channels = [64, 128, 256, 512], stride_stage =1, 
                 kernel_size=3, invert=False, 
                 layer_scale_init_value=1e-6 ):

        block_args = block_args or dict()
        assert output_stride == 32

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(Fedconv, self).__init__()

        # Stem
        inplanes = channels[0]
        
        self.stem = nn.Sequential(
        nn.Conv2d(in_chans, inplanes//2, kernel_size=3, stride=2, padding=1, bias=True),
        _norm(norm_layer, inplanes//2),
        nn.Conv2d(inplanes//2, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
        _norm(norm_layer, inplanes),
        nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
        )
        
        stage_modules = make_blocks(
            block, channels, layers, inplanes,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, stride_stage=stride_stage, 
            kernel_size=kernel_size, invert=invert, 
            layer_scale_init_value=layer_scale_init_value, **block_args)

        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        

        # Head (Pooling and Classifier)
        self.num_features = channels[-1] * block.expansion if not invert else channels[-1]
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', _norm(norm_layer, self.num_features) if norm_layer!=nn.InstanceNorm2d else nn.Identity()),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))

        named_apply(partial(_init_weights), self)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
            # pool -> norm -> fc
        self.head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
            ('norm', self.head.norm),
            ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

    def forward_features(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        Fedconv, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)
    

@register_model
def fedconv_base(pretrained=False, **kwargs):  
    '''
    FedConv-Base
    '''
    model_args = dict(block=fedconv_block_nonnorm, layers=[3, 3, 5, 3], channels=[96, 192, 384, 768], 
                       act_layer=nn.SiLU(), kernel_size=9, 
                       **kwargs)
    return _create_resnet('fedconv', pretrained, **model_args)

@register_model
def fedconv_invert(pretrained=False, **kwargs): 
    '''
    FedConv-Invert
    '''
    model_args = dict(block=fedconv_invert_block_nonnorm, layers=[3, 3, 6, 3], channels=[96, 192, 384, 768], 
                       act_layer=nn.SiLU(), kernel_size=9, 
                      invert=True, **kwargs)
    return _create_resnet('fedconv', pretrained, **model_args)


@register_model
def fedconv_invertup(pretrained=False, **kwargs):  
    '''
    FedConv-InvertUp
    '''
    model_args = dict(block=fedconv_invertup_block_nonnorm, layers=[3, 4, 9, 3], channels=[96, 192, 384, 768], 
                       act_layer=nn.SiLU(), kernel_size=9, 
                      invert=True, **kwargs)
    return _create_resnet('fedconv', pretrained, **model_args)
