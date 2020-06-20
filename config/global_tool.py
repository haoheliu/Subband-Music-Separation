#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   global_tool.py    
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/4 5:54 PM   Haohe Liu      1.0         None
'''


from config.mainConfig import Config
from util.subband.ana_syn_util import PQMF

class GlobalTool:
    # Tools
    subband_number = Config.subband
    if(Config.split_band):
        if (Config.use_gpu):
            qmf = PQMF(subband_number, 64,project_root=Config.project_root).cuda(Config.device)
        else:
            qmf = PQMF(subband_number, 64,project_root=Config.project_root)

    @classmethod
    def refresh_subband(self,subband_number):
        if (Config.split_band):
            if (Config.use_gpu):
                GlobalTool.qmf = PQMF(subband_number, 64,project_root=Config.project_root).cuda(Config.device)
            else:
                GlobalTool.qmf = PQMF(subband_number, 64,project_root=Config.project_root)
