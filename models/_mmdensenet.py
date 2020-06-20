#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _mmdensenet.py
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/29 11:51 PM   Haohe Liu      1.0         None
'''

import torch
# import encoding ## pip install torch-encoding . For synchnonized Batch norm in pytorch 1.0.0
import torch.nn as nn
from models._mdensenet import _MDenseNet,_DenseBlock
from torch.nn import functional as F

class MMDenseNet(nn.Module):
    def __init__(self, input_channel,drop_rate=0.1):
        super(MMDenseNet, self).__init__()
        kl_low = [(14,4), (16,4), (16,4), (16,4), (16,4), (16,4), (16, 4)]
        kl_high = [(10,3), (10,3), (10,3), (10,3), (10,3), (10,3), (16, 3)]
        # kl_high = [(14,4), (14,4), (14,4), (14,4), (14,4), (14,4), (14, 4)]
        kl_full = [(6, 2), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2),]
        self.lowNet = _MDenseNet_STEM(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3, kl=kl_low, drop_rate=drop_rate, )
        self.highNet = _MDenseNet_STEM(input_channel=input_channel, first_channel=32, first_kernel=(3, 3), scale=3, kl=kl_high, drop_rate=drop_rate, )
        self.fullNet = _MDenseNet_STEM(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3, kl=kl_full, drop_rate=drop_rate, )
        last_channel = self.lowNet.channels[-1] + self.fullNet.channels[-1]
        self.out = nn.Sequential(
            _DenseBlock(
                2, last_channel, 32, drop_rate),
            nn.Conv2d(32, input_channel, 1)
        )

    def forward(self, input):
        #         print(input.shape)
        B, C, F, T = input.shape
        low_input = input[:, :, :F // 2, :]
        high_input = input[:, :, F // 2:, :]
        low = self.lowNet(low_input)
        high = self.highNet(high_input)
        output = torch.cat([low, high ], 2)
        full_output = self.fullNet(input)
        output = torch.cat([output, full_output], 1)
        output = self.out(output)
        return output

class _MDenseNet_STEM(nn.Module):
    def __init__(self, input_channel=2,
                 first_channel=32,
                 first_kernel=(3, 3),
                 scale=3,
                 kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                 drop_rate=0.1):
        super(_MDenseNet_STEM, self).__init__()
        self.first_channel = 32
        self.first_kernel = first_kernel
        self.scale = scale
        self.kl = kl
        self.first_conv = nn.Conv2d(input_channel, first_channel, first_kernel)
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.dense_padding = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.channels = [self.first_channel]
        ## [_,d1,...,ds,ds+1,u1,...,us]
        for k, l in kl[:scale + 1]:
            self.dense_layers.append(_DenseBlock(
                l, self.channels[-1], k, drop_rate))
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(k, k, kernel_size=(1, 1)),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            )
            self.channels.append(k)
        for i, (k, l) in enumerate(kl[scale + 1:]):
            self.upsample_layers.append(
                nn.ConvTranspose2d(self.channels[-1], self.channels[-1], kernel_size=2, stride=2))
            self.dense_layers.append(_DenseBlock(
                l, self.channels[-1]+self.channels[-(2+2*i)], k, drop_rate))
            self.channels.append(k)

        # self.out = nn.Sequential(
        #     _DenseBlock(
        #         2, self.channels[-1], out_growth_rate, drop_rate),
        #     nn.Conv2d(out_growth_rate, input_channel, 1)
        # )

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):
        ## stem
        output = self.first_conv(input)
        dense_outputs = []

        ## downsample way
        for i in range(self.scale):
            output = self.dense_layers[i](output)
            dense_outputs.append(output)
            output = self.downsample_layers[i](output)  ## downsample

        ## upsample way
        output = self.dense_layers[self.scale](output)
        for i in range(self.scale):
            output = self.upsample_layers[i](output)
            output = self._pad(output, dense_outputs[-(i + 1)])
            output = torch.cat((output, dense_outputs[-(i + 1)]),dim=1)
            output = self.dense_layers[self.scale + 1 + i](output)
        output = self._pad(output, input)
        return output

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# # a = torch.randn(1,16,300,100)
# model = MMDenseNet(input_channel=16)
# # print(model)
# # print(model(a).size())
# print(get_parameter_number(model))

