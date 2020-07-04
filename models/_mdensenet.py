
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _mdensenet.py
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/5/5 4:10 PM   Haohe Liu      1.0         None
'''

from torch import nn
import torch
from torch.nn import functional as F
from torch import Tensor


class MDenseNet(nn.Module):
    def __init__(self,in_channel,out_channel,first_channel=32,drop_rate = 0.1):
        super(MDenseNet,self).__init__()
        self.model = _MDenseNet(in_channel,first_channel = first_channel,drop_rate = drop_rate)
        self.out = nn.Sequential(
            _DenseBlock(
                2, 16, 32, 0.1),
            nn.Conv2d(32, out_channel, 1)
        )
    def forward(self, input):
        return self.out(self.model(input))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        prev_features = torch.cat(prev_features, 1)
        new_features = self.conv2(self.relu2(self.norm2(prev_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.num_input_features = num_input_features
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return features[-1]


class _MDenseNet(nn.Module):
    def __init__(self, input_channel=2,
                 first_channel=32,
                 first_kernel=(3, 3),
                 scale=4,
                 kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4),(16, 4), (16, 4)],
                 # kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                 drop_rate=0.1):
        super(_MDenseNet, self).__init__()
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
