#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   subband_util.py    
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/3 4:54 PM   Haohe Liu      1.0         None
'''

from util.dsp_util import stft, istft
from config.global_tool import GlobalTool
import torch

def before_forward_f(
        *args,
        device = torch.device("cpu"),
        subband_num=4,
        sample_rate=44100,
        normalize = True
):
    '''
    This function can only be used in frequency domain
    Args:
        *args: multiple raw wave inputs, format: [batchsize,channels,wave],value:[-32767,32767]
        device: torch.device
        subband_num: int, subband number to split:[2,4,8]
        sample_rate: int, default 44100
        normalize: if true the value of the output will be normalized to [-1,1]

    Returns:
        For each raw wave inputs, return a list with the same order as inputs
    '''
    def merge_channel(data):
        res = None
        for i in range(data.size(1)):
            if (res is None):
                res = data[:, i, ...]
            else:
                res = torch.cat((res, data[:, i, ...]), dim=-1)
        return res
    res = []
    for each in args: # for each keywords
        batchsize = each.size()[0]
        if(normalize):wave = each / 32768.0
        else:wave = each
        if (subband_num!=1):
            # wave = wave.unsqueeze(1) # torch.Size([3, 1, 22050])
            subband = GlobalTool.qmf.analysis(wave) # torch.Size([3, 4, 5512])
            f_subband = None
            for i in range(batchsize): # for each subband
                # torch.Size([4, 513, 20, 2])
                item_f_subband = stft(subband[i,:,:],sample_rate=sample_rate/subband_num,device=device)
                item_f_subband = item_f_subband.permute(0,3,1,2)
                size = item_f_subband.size()
                item_f_subband = item_f_subband.reshape(-1,size[2],size[3]).unsqueeze(0)
                if(f_subband is None):
                    f_subband = item_f_subband # add a dimension for batch
                else:
                    f_subband = torch.cat((f_subband,item_f_subband),dim=0)
            res.append(f_subband)
        else:
            f_subband = None
            for i in range(batchsize):
                if(f_subband is None):
                    f_subband = stft(wave[i], sample_rate=sample_rate, device=device).unsqueeze(0)
                    f_subband = merge_channel(f_subband)
                else:
                    new = stft(wave[i], sample_rate=sample_rate, device=device).unsqueeze(0)
                    new = merge_channel(new)
                    f_subband = torch.cat((f_subband,new),dim=0)
            res.append(f_subband.permute(0,3,1,2))
    if(len(res) == 1):
        return res[0]
    else:
        return res


def after_forward_f(
        *args,
        device = torch.device("cpu"),
        subband_num=4,
        sample_rate=44100,
        normalized = True
):
    '''
        This function can only be used in frequency domain
        Args:
            *args: Arbitrary number of input, format:
                if(subband):[batchsize,subband_num*2*2,frequency_bin,time_step]
                if(not subband):[batchsize,4,frequency_bin,time_step]
            device: torch.device
            subband_num: int, subband number to split:[2,4,8]
            sample_rate: int, default 44100
            normalize: if true the value of the output will be normalized to [-1,1]

        Returns:
            A list of reconstructed raw waves
        '''
    def split_channels(data):
        data = data.permute(0,2,3,1)
        res = None
        for i in range(data.size()[-1]):
            if(i%2 == 0):
                if(res is None):
                    res = data[...,i:i+2].unsqueeze(1)
                else:
                    new = data[...,i:i+2].unsqueeze(1)
                    res = torch.cat((res,new),dim=1)
        return res

    res = []
    for each in args:
        batchsize = each.size()[0]
        if(subband_num!=1):
            t_subband = None
            for i in range(batchsize):
                size = each.size()
                item_t_subband = each[i].reshape(size[1]//2,2,size[2],size[3]).permute(0,2,3,1)
                item_t_subband = istft(item_t_subband,sample_rate=sample_rate/subband_num,device = device).unsqueeze(0)
                item_t_subband = GlobalTool.qmf.synthesis(item_t_subband)
                if(t_subband is None):
                    t_subband = item_t_subband
                else:
                    t_subband = torch.cat((t_subband,item_t_subband),dim=0)
            if (normalized): t_subband *= 32768
            res.append(t_subband)
        else:
            each = split_channels(each)
            t_subband = None
            for i in range(batchsize):
                if(t_subband is None):
                    t_subband = istft(each[i],sample_rate=sample_rate,device=device).unsqueeze(0)
                else:
                    new = istft(each[i],sample_rate=sample_rate,device=device).unsqueeze(0)
                    t_subband = torch.cat((t_subband,new),dim=0)
            if (normalized): t_subband *= 32768
            res.append(t_subband)
    if (len(res) == 1):
        return res[0]
    else:
        return res

