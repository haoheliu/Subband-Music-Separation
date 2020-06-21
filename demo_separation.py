#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo_separation.py    
@Contact :   liu.8948@osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/6/21 3:53 PM   Haohe Liu      1.0         None
'''

import torch
import os
from config.mainConfig import Config
from config.global_tool import GlobalTool
from util.separation_util import SeparationUtil
from models.dedicated import dedicated_model

subband=8
GlobalTool.refresh_subband(subband)

load_model_path = "./checkpoints/1_2020_5_8_MDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs16-1_fl1.5_ss4500.0_87lnu4fshift8flength32drop0.1split_bandTrue_8"
start_point = 155700

if (Config.split_band):
    inchannels = 4 * subband
    outchannels = 4 * subband
else:
    inchannels = outchannels = 4

model = dedicated_model(model_name="MDenseNet",
                        device="cpu",
                        inchannels=inchannels,
                        outchannels=outchannels,
                        sources=2,
                        drop_rate=0.1)

model.load_state_dict(torch.load(os.path.join(load_model_path,"model"+str(start_point)+".pth")))

su = SeparationUtil(model=model,
                    device='cpu',
                    MUSDB_PATH="Your path here",
                    split_band=subband,
                    project_root='./',
                    trail_name="demo_separation")
# on musdb
# su.evaluate(save_wav=True,save_json=True)

# all .wav in "evaluate/listener_todo"
su.split_listener()
