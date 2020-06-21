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
import sys
sys.path.append("..")
from util.data_util import write_json,load_json

# Project Structure
def find_and_build(project_root,path):
    path = os.path.join(project_root, path)
    if not os.path.exists(path):
        os.mkdir(path)

class Config:

    # Data path
    MUSDB18_PATH = ""

    # Model configurations
    sources = 2
    model_name = "Unet-5"  # ["Unet-6" "MMDenseNet" "MDenseNet"]

    # Split four bands
    subband = 4

    # Validation loss decrease threshold
    decrease_ratio = 0.98

    # Reload pre-trained model
    load_model_path = ""
    start_point = 0

    # Hyper-params
    epoches = 300
    learning_rate = 0.001
    batch_size = 4
    accumulation_step = 10
    gamma = 0.95

    frame_length = 3
    drop_rate = 0.2

    # Training
    device_str = "cpu"

    # loss conponents
    loss_component = ['l1','l2','l3']



    # Additional data
    ## vocal data
    additional_vocal_data = [
    ]

    ### background data
    additional_accompaniment_data = [
    ]

    # TRAIN
    every_n = 10
    show_model_structure = True
    ##############################################################################
    # Auto generated parameters

    conf = {}
    project_root = os.getcwd() + "/"
    sample_rate = 44100
    if (model_name == "Unet-5"):
        model_name_alias = "_unet_5_"
    if (model_name == "Unet-6"):
        model_name_alias = "_unet_6_"
    elif (model_name == "MMDenseNet"):
        model_name_alias = "MMDenseNet"
    elif (model_name == "MDenseNet"):
        model_name_alias = "MDenseNet"

    num_workers = batch_size
    if (len(additional_vocal_data) != 0 or len(additional_accompaniment_data) != 0):
        BIG_DATA = True
    else:
        BIG_DATA = False

    if (BIG_DATA):
        step_size = int(180000 / batch_size)  # Every 45 h
    else:
        step_size = int(72000 / batch_size)  # Every 30 h

    if (not BIG_DATA):
        validation_interval = int(3600 / batch_size)  # Every 1.5 h
    else:
        validation_interval = int(18000 / batch_size)  # Every 4.5h

    split_band = True if subband != 1 else False

    if "cuda" in str(device_str):
        use_gpu = True
    else:
        use_gpu = False
    device = torch.device(device_str if use_gpu else "cpu")


    @classmethod
    def refresh_configuration(cls,path_to_config_json):
        conf_json = load_json(path_to_config_json)

        # Data path
        Config.MUSDB18_PATH = conf_json['PATH']['MUSDB18_PATH']

        # Model configurations
        Config.sources = conf_json['MODEL']['sources']
        Config.model_name = conf_json['MODEL']['model_name']  # ["Unet-6" "MMDenseNet" "MDenseNet"]

        # Split four bands
        Config.subband = conf_json['SUBBAND']['number']

        # Validation loss decrease threshold
        Config.decrease_ratio = conf_json["VALIDATION"]['decrease_ratio']

        # Reload pre-trained model
        Config.load_model_path = conf_json['MODEL']['PRE-TRAINED']['load_model_path']
        Config.start_point = conf_json['MODEL']['PRE-TRAINED']['start_point']


        # Hyper-params
        Config.epoches = conf_json["TRAIN"]['epoches']
        Config.learning_rate = conf_json["TRAIN"]['learning_rate']['initial']
        Config.batch_size = conf_json["TRAIN"]['batchsize']
        Config.accumulation_step = conf_json["TRAIN"]['accumulation_step']
        Config.gamma = conf_json["TRAIN"]['learning_rate']['gamma_decrease']

        Config.frame_length = conf_json["TRAIN"]['frame_length']
        Config.drop_rate = conf_json["TRAIN"]['dropout']

        # Training
        Config.device_str = conf_json["TRAIN"]['device_str']

        # loss conponents
        Config.loss_component = conf_json["TRAIN"]['loss']

        # Additional data
        ## vocal data
        Config.additional_vocal_data = conf_json["PATH"]['additional_data']["additional_vocal_path"]

        ### background data
        Config.additional_accompaniment_data = conf_json["PATH"]['additional_data']["additional_accompaniments_path"]


        # TRAIN
        Config.every_n = conf_json["LOG"]["every_n"]
        Config.show_model_structure = True if conf_json["LOG"]["show_model_structure"] == 1 else False
    ##############################################################################
        # Auto generated parameters

        Config.conf = {}
        Config.project_root = os.getcwd()+"/"
        Config.sample_rate = 44100
        if (Config.model_name == "Unet-5"):
            Config.model_name_alias = "_unet_5_"
        if (Config.model_name == "Unet-6"):
            Config.model_name_alias = "_unet_6_"
        elif (Config.model_name == "MMDenseNet"):
            Config.model_name_alias = "MMDenseNet"
        elif (Config.model_name == "MDenseNet"):
            Config.model_name_alias = "MDenseNet"

        Config.num_workers = Config.batch_size
        if(len(Config.additional_vocal_data)!=0 or len(Config.additional_accompaniment_data)!=0):
            Config.BIG_DATA = True
        else:
            Config.BIG_DATA = False

        if (Config.BIG_DATA):
            Config.step_size = int(180000 / Config.batch_size)  # Every 45 h
        else:
            Config.step_size = int(72000 / Config.batch_size)  # Every 30 h

        if (not Config.BIG_DATA):
            Config.validation_interval = int(3600 / Config.batch_size)  # Every 1.5 h
        else:
            Config.validation_interval = int(18000 / Config.batch_size)  # Every 4.5h

        Config.split_band = True if Config.subband!=1 else False

        if "cuda" in str(Config.device_str):
            Config.use_gpu = True
        else:
            Config.use_gpu = False
        Config.device = torch.device(Config.device_str if Config.use_gpu else "cpu")

        # Build trail name
        cur = datetime.datetime.now()
        Config.trail_name = str(cur.year) + "_" + str(cur.month) + "_" + str(
            cur.day) + "_" + Config.model_name_alias + "sf" + str(Config.start_point) + "_"
        Config.counter = 1
        for each in os.listdir(Config.project_root + "saved_models"):
            t = str(cur.year) + "_" + str(cur.month) + "_" + str(cur.day)
            if (t in each):
                for dirName in os.listdir(Config.project_root + "saved_models/" + each):
                    if ("model" in dirName):
                        Config.counter += 1
                        break


        Config.trail_name = str(Config.counter) + "_" + Config.trail_name
        for each in Config.loss_component:
            Config.trail_name += each + "_"
        Config.trail_name.strip("_")
        Config.trail_name += "_BD_" + str(Config.BIG_DATA) + "_lr" + str(Config.learning_rate).split(".")[-1] + "_" \
                      + "bs" + str(Config.batch_size) + "-" + str(Config.accumulation_step) + "_" \
                      + "fl" + str(Config.frame_length) + "_" \
                      + "ss" + str(Config.step_size) + "_" + str(Config.gamma).split(".")[-1] \
                      + "drop" + str(Config.drop_rate) \
                      + "split_band" + str(Config.split_band) + "_" + str(Config.subband)
        # +"emptyN"+str(empty_every_n)\
        print("Write config file at: ", Config.project_root + "config/json/" + Config.trail_name + ".json")
        write_json(Config.conf, Config.project_root + "config/json/" + Config.trail_name + ".json")

        Config.conf['model_name'] = Config.model_name
        Config.conf['split_band'] = Config.split_band
        Config.conf['decrease_ratio'] = Config.decrease_ratio
        Config.conf['start_point'] = Config.start_point
        Config.conf['learning_rate'] = Config.learning_rate
        Config.conf['batch_size'] = Config.batch_size
        Config.conf['accumulation_step'] = Config.accumulation_step
        Config.conf['step_size'] = Config.step_size
        Config.conf['gamma'] = Config.gamma
        Config.conf['sample_rate'] = Config.sample_rate
        Config.conf['frame_length'] = Config.frame_length
        Config.conf['drop_rate'] = Config.drop_rate

        find_and_build(Config.project_root,"outputs")
        find_and_build(Config.project_root,"outputs/listener")
        find_and_build(Config.project_root,"outputs/musdb_test")
        find_and_build(Config.project_root,"saved_model")
        find_and_build(Config.project_root,"config/json")
        find_and_build(Config.project_root,"evaluate/listener_todo")

