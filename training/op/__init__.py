"""
    Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/op/__init__.py
"""
from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d