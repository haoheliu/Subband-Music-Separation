import sys

sys.path.append("..")
from models.dedicated import dedicated_model
from dataloader import dataloader
from torch.utils.data import DataLoader
from util.subband.subband_util import before_forward_f
import torch
import time
from util.separation_util import SeparationUtil
import logging
import os
from config.mainConfig import Config
from config.global_tool import GlobalTool

if(len(sys.argv) <= 1):
    raise ValueError("Error: You must specify a config file, example: python xxx.py config_xxx.json")

Config.refresh_configuration(sys.argv[1])
GlobalTool.refresh_subband(Config.subband)

if (not os.path.exists(Config.project_root + "saved_models/" + Config.trail_name)):
    os.mkdir(Config.project_root + "saved_models/" + Config.trail_name + "/")
    print("MakeDir: " + Config.project_root + "saved_models/" + Config.trail_name)

# Cache for data
freq_bac_loss_cache = []
freq_voc_loss_cache = []
freq_cons_loss_cache = []
validate_score = (None, None)

loss = torch.nn.L1Loss()

if (Config.split_band):
    inchannels = 4 * Config.subband
    outchannels = 4 * Config.subband
else:
    inchannels = outchannels = 4
model = dedicated_model(model_name=Config.model_name,
                        device=Config.device,
                        inchannels=inchannels,
                        outchannels=outchannels,
                        sources=2,
                        drop_rate=Config.drop_rate)
if (Config.use_gpu):
    model = model.cuda(Config.device)

# MODEL
if (not Config.start_point == 0):
    print("Load model from ", Config.load_model_path + "/model" + str(Config.start_point) + ".pth")
    model.load_state_dict(torch.load(Config.load_model_path + "/model" + str(Config.start_point) + ".pth"))
    model.cnt = Config.start_point

if(Config.show_model_structure):
    print(model)

print("Start training from ", model.cnt, Config.model_name)

# DATALOADER
dl = torch.utils.data.DataLoader(
    dataloader.WavenetDataloader(frame_length=Config.frame_length,
                                 sample_rate=Config.sample_rate,
                                 num_worker=Config.num_workers,
                                 MUSDB18_PATH=Config.MUSDB18_PATH,
                                 BIG_DATA=Config.BIG_DATA,
                                 additional_background_data=Config.additional_accompaniment_data,
                                 additional_vocal_data=Config.additional_vocal_data
                                 ),
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=Config.num_workers)

# OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
optimizer.zero_grad()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)


def save_and_validate():
    global validate_score
    su = SeparationUtil(model=model,
                        device=Config.device,
                        MUSDB_PATH=Config.MUSDB18_PATH,
                        split_band=Config.subband,
                        sample_rate=Config.sample_rate,
                        project_root=Config.project_root,
                        trail_name=Config.trail_name)
    print("Start validation process...")
    try:
        validate_score = su.validate(validate_score)
    except Exception as e:
        print("Error occured while evaluating...")
        logging.exception(e)
    del su


def train(  # Time Domain
        target_background,
        target_vocal,
        target_song,
):
    def pre_pro(tensor: torch.Tensor):
        # move channel axis to the second dimension
        return tensor.permute(0, 2, 1).float()

    target_background, target_vocal, target_song = pre_pro(target_background), pre_pro(target_vocal), pre_pro(
        target_song)

    gt_bac, gt_voc, gt_song = before_forward_f(target_background, target_vocal, target_song,
                                               subband_num=Config.subband,
                                               device=Config.device,
                                               sample_rate=Config.sample_rate,
                                               normalize=False)

    if ('l1' in Config.loss_component):
        output_track = []
        for track_i in range(Config.sources):
            mask = model(track_i, gt_song)
            out = mask * gt_song
            output_track.append(out)
            # All tracks is done
            if (track_i == Config.sources - 1):
                # Preprocessing
                output_track_sum = sum(output_track)
                output_background = output_track[0]
                output_vocal = output_track[1]
                # Calculate loss function
                ## conservation loss
                lossVal = loss(output_track_sum, gt_song) / Config.accumulation_step
                freq_cons_loss_cache.append(float(lossVal) * Config.accumulation_step)
                ## l1 loss (accompaniment)
                temp2 = loss(output_background, gt_bac) / Config.accumulation_step
                lossVal += temp2
                freq_bac_loss_cache.append(float(temp2) * Config.accumulation_step)
                ## l1 loss (vocal)
                temp3 = loss(output_vocal, gt_voc) / Config.accumulation_step
                lossVal += temp3
                freq_voc_loss_cache.append(float(temp3) * Config.accumulation_step)
        # Backward
        lossVal.backward()
        if (model.cnt % Config.accumulation_step == 0 and model.cnt != Config.start_point):
            # Optimize
            optimizer.step()
            optimizer.zero_grad()
    else:
        freq_bac_loss, freq_voc_loss = 0, 0
        # An momory efficient version
        for track_i in range(Config.sources):
            mask = model(track_i, gt_song)
            out = mask * gt_song
            if (track_i == 1):
                lossVal = loss(out, gt_voc)
                freq_voc_loss = float(lossVal)
            else:
                lossVal = loss(out, gt_bac)
                freq_bac_loss = float(lossVal)
            # Backward
            lossVal.backward()
            # Optimize
            optimizer.step()
            optimizer.zero_grad()
        freq_bac_loss_cache.append(freq_bac_loss)
        freq_voc_loss_cache.append(freq_voc_loss)


t0 = time.time()
for epoch in range(Config.epoches):
    print("EPOCH: ", epoch)
    start = time.time()
    if (Config.use_gpu):
        pref = dataloader.data_prefetcher(dl, device=Config.device)
        background, vocal, song, name = pref.next()
        while background is not None:
            if model.cnt % Config.validation_interval == 0 and model.cnt != Config.start_point:
                save_and_validate()
            if model.cnt % Config.every_n == 0 and model.cnt != Config.start_point:
                t1 = time.time()
                print(str(model.cnt) +
                      " Freq L1loss voc",
                      format((sum(freq_voc_loss_cache[-Config.every_n:]) / Config.every_n) * 10000, '.7f'),
                      " Freq L1loss bac",
                      format((sum(freq_bac_loss_cache[-Config.every_n:]) / Config.every_n) * 10000, '.7f'),
                      " Freq conserv-loss",
                      format((sum(freq_cons_loss_cache[-Config.every_n:]) / Config.every_n) * 10000, '.7f'),
                      " lr:", optimizer.param_groups[0]['lr'],
                      " speed:", format((Config.frame_length * Config.batch_size) / (t1 - t0), '.2f'))
                freq_voc_loss_cache = []
                freq_bac_loss_cache = []
                freq_cons_loss_cache = []
            t0 = time.time()
            train(
                target_background=background,
                target_vocal=vocal,
                target_song=song)
            background, vocal, song, name = pref.next()
            if model.cnt > 100:
                scheduler.step(1)
            model.cnt += 1
        end = time.time()
        print("Epoch " + str(epoch) + " finish, total time: " + str(end - start))
    else:
        for background, vocal, song, name in dl:
            if model.cnt % Config.validation_interval == 0 and model.cnt != Config.start_point:
                save_and_validate()
            if model.cnt % Config.every_n == 0 and model.cnt != Config.start_point:
                t1 = time.time()
                print(str(model.cnt) +
                      " Freq L1loss voc",
                      format((sum(freq_voc_loss_cache[-Config.every_n:]) / Config.every_n) * 10000, '.7f'),
                      " Freq L1loss bac",
                      format((sum(freq_bac_loss_cache[-Config.every_n:]) / Config.every_n) * 10000, '.7f'),
                      " Freq conserv-loss",
                      format((sum(freq_cons_loss_cache[-Config.every_n:]) / Config.every_n) * 10000, '.7f'),
                      " lr:", optimizer.param_groups[0]['lr'],
                      " speed:", format((Config.frame_length * Config.batch_size) / (t1 - t0), '.2f'))
                freq_voc_loss_cache = []
                freq_bac_loss_cache = []
                freq_cons_loss_cache = []
            t0 = time.time()
            train(
                target_background=background,
                target_vocal=vocal,
                target_song=song)
            if model.cnt > 100:
                scheduler.step(1)
            model.cnt += 1
        end = time.time()
        print("Epoch " + str(epoch) + " finish, total time: " + str(end - start))
