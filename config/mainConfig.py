#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   small_denseunet.py    
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/10 7:50 PM   Haohe Liu      1.0         None
'''
import torch
import os
import datetime
import json
from util.data_util import write_json,load_json

class Config:
    conf = {}
    # Run mode
    # Project configurations
    project_root = "/home/work_nfs/hhliu/workspace/github/subband-unet/"
    datahub_root = "/home/work_nfs/hhliu/workspace/datasets/"
    MUSDB18_PATH = "/home/work_nfs/hhliu/workspace/datasets/musdb18hq_splited"
    # Model configurations
    channels = 2
    model_name = "MDenseNet"  # "Unet-6" "MMDenseNet" "MDenseNet"

    if (model_name == "Unet-5"):
        model_name_alias = "_unet_5_"
    if (model_name == "Unet-6"):
        model_name_alias = "_unet_6_"
    elif(model_name == "MMDenseNet"):
        model_name_alias = "MMDenseNet"
    elif (model_name == "MDenseNet"):
        model_name_alias = "MDenseNet"

    # Split four bands
    split_band = True
    subband = 1 if(not split_band) else 4
    decrease_ratio = 0.98
    BIG_DATA = False
    # Reload pre-trained model
    # 167400
    # load_model_path = project_root+"saved_models/1_2020_4_6_DenseUnet_4_4_4_12_0.2_spleeter_sf0_l1_l2_l3_lr0005_bs2-15_fl1.5_ss64000_85lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue"
    load_model_path = project_root + "saved_models/1_2020_5_23_MDenseNetspleeter_sf637200_l1_l2_l3__BD_True_lr0001_bs4-1_fl3_ss45000_97lnu4fshift8flength32drop0.2split_bandTrue_4"
    start_point = 0

    # Hyper-params
    epoches = 3000
    learning_rate = 0.001
    batch_size = 4
    accumulation_step = 1
    if(BIG_DATA):step_size = int(180000/batch_size) # Every 45 h
    else:step_size = int(72000/batch_size) # Every 30 h
    gamma = 0.97
    sample_rate = 44100
    num_workers = batch_size
    frame_length = 1.5
    # empty_every_n = 5
    drop_rate = 0.2

    # Training
    use_gpu = True
    device_str = "cuda:2" #todo
    device = torch.device(device_str if use_gpu else "cpu")
    best_sdr_vocal, best_sdr_background = None, None
    if(not BIG_DATA):
        validation_interval = int(3600/batch_size) # Every 1.5 h
    else:
        validation_interval = int(18000 / batch_size) # Every 4.5h

    conf['model_name'] = model_name
    conf['split_band'] = split_band
    conf['decrease_ratio'] = decrease_ratio
    conf['start_point'] = start_point
    conf['learning_rate'] = learning_rate
    conf['batch_size'] = batch_size
    conf['accumulation_step'] = accumulation_step
    conf['step_size'] = step_size
    conf['gamma'] = gamma
    conf['sample_rate'] = sample_rate
    conf['frame_length'] = frame_length
    conf['drop_rate'] = drop_rate

    '''
    l1: Frequency domain energy conservation l1 loss
    l2: Frequency domain l1 loss on background
    l3: Frequency domain l1 loss on vocal 
    Your option: l2,l3 or l1,l2,l3
    '''

    loss_component = [
        # 'l1',
        'l2',
        'l3',
    ]

    if ('l4' in loss_component
            or 'l5' in loss_component
            or 'l6' in loss_component
            or 'l7' in loss_component
            or 'l8' in loss_component
            ):
        time_domain_loss = True
    else:
        time_domain_loss = False

    # Build trail name
    cur = datetime.datetime.now()
    trail_name = str(cur.year) + "_" + str(cur.month) + "_" + str(
        cur.day) + "_" + model_name_alias + "sf" + str(start_point) + "_"
    counter = 1
    for each in os.listdir(project_root + "saved_models"):
        t = str(cur.year) + "_" + str(cur.month) + "_" + str(cur.day)
        if (t in each):
            for dirName in os.listdir(project_root + "saved_models/" + each):
                if ("model" in dirName):
                    counter += 1
                    break

    trail_name = str(counter) + "_" + trail_name
    for each in loss_component:
        trail_name += each + "_"
    trail_name.strip("_")
    trail_name += "_BD_"+str(BIG_DATA)+"_lr" + str(learning_rate).split(".")[-1] + "_" \
                  + "bs" + str(batch_size) + "-" + str(accumulation_step) + "_" \
                  + "fl" + str(frame_length) + "_" \
                  + "ss" + str(step_size) + "_" + str(gamma).split(".")[-1] \
                  + "drop" + str(drop_rate) \
                  + "split_band" + str(split_band) +"_"+str(subband)
    # +"emptyN"+str(empty_every_n)\
    print("Write config file at: ",project_root+"config/json/"+trail_name+".json")
    write_json(conf,project_root+"config/json/"+trail_name+".json")

# Additional data
    ## vocal data
    vocal_data = [
    ]

    ### background data
    background_data = [
    ]


    # Others
    print_model_struct = False