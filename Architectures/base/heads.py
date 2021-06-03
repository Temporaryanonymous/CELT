import torch
import torch.nn as nn
from .modules import Flatten, Activation


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

class CELT_SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super(CELT_SegmentationHead, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)
        self.conv_att_to_all_1 = nn.Conv2d(1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        #self.conv_att_to_all_2 = nn.Conv2d(64,out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        #self.conv_att_to_all_3 = nn.Conv2d(128, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self,decoder_output,att_all):
        decoder_output=self.conv2d(decoder_output)
        decoder_output=self.upsampling(decoder_output)
        att_all = self.conv_att_to_all_1(att_all)

        #att_all = self.conv_att_to_all_2(att_all)
        #att_all = self.conv_att_to_all_3(att_all)

        #decoder_output = torch.mul(decoder_output, att_all)
        decoder_output = decoder_output+att_all

        decoder_output=self.activation(decoder_output)

        return decoder_output

class SegmentationHead_en_to_seg(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super(SegmentationHead_en_to_seg, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)

    def forward(self,decoder_output,att_all):
        decoder_output=self.conv2d(decoder_output)
        decoder_output=self.upsampling(decoder_output)

        decoder_output=torch.mul(decoder_output,att_all)
        decoder_output=self.activation(decoder_output)
        return decoder_output

