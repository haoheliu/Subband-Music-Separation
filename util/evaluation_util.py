# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
# '''
# @File    :   evaluation_util.py
# @Contact :   liu.8948@buckeyemail.osu.edu
# @License :   (C)Copyright 2020-2021
#
# @Modify Time      @Author    @Version    @Desciption
# ------------      -------    --------    -----------
# 2020/4/13 7:42 PM   Haohe Liu      1.0         None
# '''
# import os
# import sys
#
# sys.path.append("..")
# from util.separation_util import SeparationUtil
# from config.global_tool import GlobalTool
# import logging
# import torch
# final_models = {
#     # UNET-5
#     # "UNET-5": {
#     #     "alias": "UNET-5",
#     #     "path": "1_2020_5_7_MDenseNetspleeter_sf167400_l1_l2_l3__BD_False_lr0002_bs2-1_fl3_ss36000.0_87lnu4fshift8flength32drop0.1split_bandFalse_1",
#     #     "start_points": [295200],
#     #     "subband": 1,
#     # },
#     # # UNET-5 K=2
#     # "UNET-5_K=2": {
#     #     "alias": "UNET-5_K=2",
#     #     "path": "1_2020_5_6__unet_2conv_spleeter_sf0_l1_l2_l3__BD_False_lr001_bs2-1_fl3_ss36000.0_87lnu4fshift8flength32drop0.1split_bandTrue_2",
#     #     "start_points": [351000],
#     #     "subband": 2,
#     # },
#     # # UNET-5 K=4
#     # "UNET-5_K=4": {
#     #     "alias": "UNET-5_K=4",
#     #     "path": "3_2020_5_3__unet_1conv_spleeter_sf0_l1_l2_l3__BD_False_lr0006_bs4-1_fl3_ss18000.0_87lnu4fshift8flength32drop0.1split_bandTrue_4",
#     #     "start_points": [632700],
#     #     "subband": 4,
#     # },
#     # # UNET-5 K=8
#     # "UNET-5_K=8": {
#     #     "alias": "UNET-5_K=8",
#     #     "path": "3_2020_5_8__unet_2conv_spleeter_sf0_l1_l2_l3__BD_False_lr001_bs8-1_fl3_ss9000.0_87lnu4fshift8flength32drop0.1split_bandTrue_8",
#     #     "start_points": [117900],
#     #     "subband": 8,
#     # },
#     # # MMDN
#     # "MMDN": {
#     #     "alias": "MMDN",
#     #     "path": "1_2020_5_5_MMDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs4-1_fl1.5_ss18000.0_87lnu4fshift8flength32drop0.1split_bandFalse_1",
#     #     "start_points": [171900],
#     #     "subband": 1,
#     # },
#     # # MDN
#     # "MDN": {
#     #     "alias": "MDN",
#     #     "path": "1_2020_5_5_MMDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs4-1_fl1.5_ss18000.0_87lnu4fshift8flength32drop0.1split_bandFalse_1",
#     #     "start_points": [193500],
#     #     "subband": 1,
#     # },
#     # # MDN K=2
#     # "MDN_K=2": {
#     #     "alias": "MDN_K=2",
#     #     "path": "1_2020_5_8_MDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs8-1_fl1.5_ss9000.0_87lnu4fshift8flength32drop0.1split_bandTrue_2",
#     #     "start_points": [120600],
#     #     "subband": 2,
#     # },
#     # # MDN K=4
#     # "MDN_K=4": {
#     #     "alias": "MDN_K=4",
#     #     "path": "1_2020_5_5_MDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs4-1_fl1.5_ss18000.0_87lnu4fshift8flength32drop0.1split_bandTrue_4",
#     #     "start_points": [552600],
#     #     "subband": 4,
#     # },
#     # # MDN K=8
#     # "MDN_K=8": {
#     #     "alias": "MDN_K=8",
#     #     "path": "1_2020_5_8_MDenseNetspleeter_sf0_l1_l2_l3__BD_False_lr001_bs16-1_fl1.5_ss4500.0_87lnu4fshift8flength32drop0.1split_bandTrue_8",
#     #     "start_points": [155700],
#     #     "subband": 8,
#     # },
#     # # UNET-6
#     # "UNET-6": {
#     #     "alias": "UNET-6",
#     #     "path": "2_2020_4_27__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs1-1_fl3_ss72000.0_9lnu5fshift8flength32drop0.1split_bandFalse_1",
#     #     "start_points": [640800],
#     #     "subband": 1,
#     # },
#     # # UNET-6 K=2
#     # "UNET-6_K=2": {
#     #     "alias": "UNET-6_K=2",
#     #     "path": "1_2020_4_28__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs2-1_fl3_ss36000.0_9lnu5fshift8flength32drop0.1split_bandTrue_2",
#     #     "start_points": [360000],
#     #     "subband": 2,
#     # },
#     # # UNET-6 K=4
#     # "UNET-6_K=4": {
#     #     "alias": "UNET-6_K=4",
#     #     "path": "1_2020_4_27__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs4-1_fl3_ss18000.0_9lnu5fshift8flength32drop0.1split_bandTrue_4",
#     #     "start_points": [336600],
#     #     "subband": 4,
#     # },
#     # # UNET-6 K=8
#     # "UNET-6_K=8": {
#     #     "alias": "UNET-6_K=8",
#     #     "path": "5_2020_4_27__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs6-1_fl3_ss12000.0_9lnu5fshift8flength32drop0.1split_bandTrue_8",
#     #     "start_points": [225600],
#     #     "subband": 8,
#     # },
#     # # BD-UNET-6
#     # "BD-UNET-6": {
#     #     "alias": "BD-UNET-6",
#     #     "path": "1_2020_4_29__unet_spleeter_spleeter_sf338400_l1_l2_l3__BD_True_lr001_bs1-1_fl3_ss72000.0_9lnu5fshift8flength32drop0.1split_bandFalse_1",
#     #     "start_points": [648000],
#     #     "subband": 1,
#     # },
#     # # BD-UNET-6 K=2
#     # "BD-UNET-6_K=2": {
#     #     "alias": "BD-UNET-6_K=2",
#     #     "path": "1_2020_4_29__unet_spleeter_spleeter_sf187200_l1_l2_l3__BD_True_lr001_bs2-1_fl3_ss36000.0_9lnu5fshift8flength32drop0.1split_bandTrue_2",
#     #     "start_points": [599400],
#     #     "subband": 2,
#     # },
#     # BD-UNET-6 K=4
#     # "BD-UNET-6_K=4": {
#     #     "alias": "BD-UNET-6_K=4",
#     #     "path": "1_2020_4_30__unet_spleeter_spleeter_sf336600_l1_l2_l3__BD_True_lr001_bs4-1_fl3_ss18000.0_9lnu5fshift8flength32drop0.1split_bandTrue_4",
#     #     "start_points": [745200],
#     #     "subband": 4,
#     # },
#     # BD-UNET-6 K=8
#     # "BD-UNET-6_K=8": {
#     #     "alias": "BD-UNET-6_K=8",
#     #     "path": "1_2020_5_4_MMDenseNetspleeter_sf209700_l1_l2_l3__BD_True_lr001_bs4-1_fl3_ss27000.0_87lnu4fshift8flength32drop0.1split_bandTrue_4",
#     #     "start_points": [610200],
#     #     "subband": 4,
#     # },
#
#     ###############################
#     # "BBD-MDN_K=4":{
#     #     "alias": "BBD-MDN_K=4",
#     #     "path": "1_2020_5_20_MDenseNetspleeter_sf0_l1_l2_l3__BD_True_lr0005_bs10-1_fl1.5_ss18000_97lnu5fshift8flength32drop0.1split_bandTrue_4",
#     #     "start_points": [330480],
#     #     "subband": 4,
#     # },
#     # "BBD-UNET_K=4": {
#     #     "alias": "BBD-UNET_K=4",
#     #     "path": "1_2020_5_20__unet_2conv_spleeter_sf0_l1_l2_l3__BD_True_lr0005_bs8-1_fl1.5_ss22500_97lnu4fshift8flength32drop0.2split_bandTrue_4",
#     #     "start_points": [911250],
#     #     "subband": 4,
#     # },#
#     # "BBD-MDN_K=4_v2": {
#     #     "alias": "BBD-UNET_K=4",
#     #     "path": "1_2020_5_23_MDenseNetspleeter_sf637200_l1_l2_l3__BD_True_lr0001_bs4-1_fl3_ss45000_97lnu4fshift8flength32drop0.2split_bandTrue_4",
#     #     "start_points": [1296000],
#     #     "subband": 4,
#     # },
#     # 945000
#     # "BBD-MDN_K=4_v3": {
#     #     "alias": "BBD-MDN_K=4",
#     #     "path": "1_2020_5_27_MDenseNetspleeter_sf1296000_l1_l2_l3__BD_True_lr0003_bs4-1_fl3_ss45000_97lnu4fshift8flength32drop0.2split_bandTrue_4",
#     #     "start_points": [2223000],
#     #     "subband": 4,
#     # },
#     "BBD-UNET_K=4_v3": {
#         "alias": "BBD-UNET_K=4",
#         "path": "1_2020_5_27__unet_2conv_spleeter_sf610200_l1_l2_l3__BD_True_lr0001_bs4-1_fl3_ss45000_97lnu4fshift8flength32drop0split_bandTrue_4",
#         "start_points": [1179000],
#         "subband": 4,
#     },
#
#     # "BBD-MMDN": {
#     #     "alias": "BBD-MMDN",
#     #     "path": "1_2020_6_10_MMDenseNetspleeter_sf0_l1_l2_l3__BD_True_lr001_bs4-1_fl1.5_ss45000_97lnu4fshift8flength32drop0.2split_bandFalse_1",
#     #     "start_points": [540000],
#     #     "subband": 1,
#     # }
#
# }
#
#
#
#
# class EvaluationHelper:
#     def __init__(self,device=None,subband_num=None,project_root=None):
#         self.device = device
#         self.subband_num = subband_num
#         self.project_root = project_root
#
#     def evaluate(self,path,start_point,
#                  MUSDB_PATH,
#                  project_root,
#                  device = torch.device('cpu'),
#                  trail_name = "temp",
#                  save_wav = True,
#                  save_json = True,
#                  subband = 1,
#                  split_musdb = True,
#                  split_listener = True):
#
#         GlobalTool.refresh_subband(subband)
#         su = SeparationUtil(load_model_pth=path,
#                             start_point=start_point,
#                             device=device,
#                             split_band = subband,
#                             project_root=project_root,
#                             trail_name=trail_name,
#                             MUSDB_PATH=MUSDB_PATH)
#         if(split_musdb):su.evaluate(save_wav=save_wav, save_json=save_json)
#         if(split_listener):su.Split_listener()
#         del su
#
#     def batch_evaluation_subband(self,path:str,start_points:list,
#                                 save_wav = True,
#                                 save_json = True,
#                                 split_musdb = True,
#                                 split_listener = True,
#                                 test_mode = False,
#                                 subband = 4):
#         for start in start_points:
#             self.evaluate(path,start,
#                           save_wav = save_wav,
#                           save_json=save_json,
#
#                           split_musdb=split_musdb,
#                           split_listener = split_listener,
#                           subband = subband)
#
#     def main(self,project_root):
#         test = final_models
#         for each in list(test.keys()):
#             path, start_points, subband = test[each]["path"], test[each]["start_points"], test[each]["subband"]
#             for start in start_points:
#                 if not os.path.exists(Config.project_root + "saved_models/" + path + "/model" + str(start) + ".pkl"):
#                     raise ValueError(start_points, path, "none exist")
#         print("Found all models success")
#
#         for each in list(test.keys()):
#             path, start_points, subband = test[each]["path"], test[each]["start_points"], test[each]["subband"]
#             try:
#                 self.batch_evaluation_subband(Config.project_root + "saved_models/" + path, start_points,
#                                             save_wav=True,
#                                             save_json=True,
#                                             test_mode=False,
#                                             split_musdb=False,
#                                             split_listener=True,
#                                             subband=subband)
#             except Exception as e:
#                 print("error...")
#                 logging.exception(e)
#                 continue
#
