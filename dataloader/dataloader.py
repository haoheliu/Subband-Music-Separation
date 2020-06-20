import sys

sys.path.append("..")
from util.wave_util import WaveHandler
from torch.utils.data import Dataset
# These part should below 'import util'
import torch
import random
import numpy as np
import time
import musdb
import librosa


class data_prefetcher():
    def __init__(self, loader,device = torch.device('cuda')):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_background, self.next_vocal, self.next_song, self.next_name = next(self.loader)
        except StopIteration:
            self.next_background = None
            self.next_vocal = None
            self.next_song = None
            self.next_name = None
            return
        with torch.cuda.stream(self.stream):
            self.next_background = self.next_background.cuda(self.device, non_blocking=True)
            self.next_vocal = self.next_vocal.cuda(self.device, non_blocking=True)
            self.next_song = self.next_song.cuda(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        background = self.next_background
        vocal = self.next_vocal
        song = self.next_song
        name = self.next_name
        if background is not None:
            background.record_stream(torch.cuda.current_stream(self.device))
        if vocal is not None:
            vocal.record_stream(torch.cuda.current_stream(self.device))
        if song is not None:
            song.record_stream(torch.cuda.current_stream(self.device))
        self.preload()
        return background, vocal, song, name


class WavenetDataloader(Dataset):
    def __init__(self,
                 frame_length=3,
                 sample_rate=44100,
                 num_worker=1,
                 MUSDB18_PATH="",
                 BIG_DATA=False,
                 additional_background_data = [],
                 additional_vocal_data = [],
                 ):
        np.random.seed(1)
        self.sample_rate = sample_rate
        self.wh = WaveHandler()
        self.BIG_DATA = BIG_DATA
        self.music_folders = []
        for each in additional_background_data:
            self.music_folders += self.readList(each)
        self.vocal_folders = []
        for each in additional_vocal_data:
            self.vocal_folders += self.readList(each)
        self.frame_length = frame_length
        self.bac_file_num = len(self.music_folders)
        self.voc_file_num = len(self.vocal_folders)

        self.num_worker = num_worker
        self.mus = musdb.DB(MUSDB18_PATH, is_wav=True, subsets='train')
        self.pitch_shift_high = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        self.pitch_shift_low = [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0]

    def random_trunk(self):
        track = random.choice(self.mus.tracks)
        while (track.name == "Alexander Ross - Goodbye Bolero"):
            track = random.choice(self.mus.tracks)
        track.chunk_duration = self.frame_length
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        return track

    def random_bac_trunk(self):
        fname = self.music_folders[np.random.randint(0, self.bac_file_num)]
        return self.wh.random_chunk(fname, self.frame_length, normalize=True), fname

    def random_voc_trunk(self):
        fname = self.vocal_folders[np.random.randint(0, self.voc_file_num)]
        return self.wh.random_chunk(fname, self.frame_length, normalize=True), fname

    def switch_pitch_high(self, vocal):
        shift = np.random.choice(self.pitch_shift_high)
        p_vocal = np.zeros(shape=vocal.shape, dtype=np.float)
        # todo we assume all data are not mono
        p_vocal[:, 0] = librosa.effects.pitch_shift(vocal[:, 0].astype(np.float), sr=self.sample_rate, n_steps=shift)
        p_vocal[:, 1] = librosa.effects.pitch_shift(vocal[:, 1].astype(np.float), sr=self.sample_rate, n_steps=shift)
        return p_vocal

    def switch_pitch_low(self, vocal):
        shift = np.random.choice(self.pitch_shift_low)
        p_vocal = np.zeros(shape=vocal.shape)
        # todo we assume all data are not mono
        # start = time.time()
        p_vocal[:, 0] = librosa.effects.pitch_shift(vocal[:, 0].astype(np.float), sr=self.sample_rate, n_steps=shift)
        p_vocal[:, 1] = librosa.effects.pitch_shift(vocal[:, 1].astype(np.float), sr=self.sample_rate, n_steps=shift)
        end = time.time()
        return p_vocal

    def generate_chorus(self, vocal):
        coin = np.random.random()
        if (coin < 0.4):
            protion = 0.3 + coin
            return protion * vocal + (1 - protion) * self.switch_pitch_high(vocal)
        elif (coin < 0.8):
            protion = (coin - 0.4) + 0.3
            return protion * vocal + (1 - protion) * self.switch_pitch_low(vocal)
        else:
            portion = (coin - 0.8) + 0.3
            portion_chorus = (1 - portion) / 2
            return portion * vocal + portion_chorus * self.switch_pitch_low(
                vocal) + portion_chorus * self.switch_pitch_high(vocal)

    def get_upper(self):
        return np.random.random() * 0.2 + 0.3

    def unify_energy(self, audio):
        upper = 0.4
        val_max = np.max(audio)
        if (val_max < 0.001):
            return audio
        else:
            return audio * (upper / val_max)

    def __getitem__(self, item):
        if (self.BIG_DATA):
            dice = np.random.random()
        else:
            dice = -1
        if (dice == -1 or dice < 0.05):
            keys = ['bass', 'drums', 'other', 'accompaniment']
            track_bac = self.random_trunk()
            track_voc = self.random_trunk()
            bac_target = random.choice(keys)
            b = self.unify_energy(track_bac.targets[bac_target].audio)
            v = self.unify_energy(track_voc.targets['vocals'].audio)
            if (dice < 0.02):
                v = self.generate_chorus(v)
            return b, v, b + v, (bac_target + "-" + track_bac.name, "vocals-" + track_voc.name)
        else:
            track_bac, name_bac = self.random_bac_trunk()
            track_voc, name_voc = self.random_voc_trunk()
            track_voc, track_bac = self.unify_energy(track_voc), self.unify_energy(track_bac)
            if (dice < 0.45):
                track_voc = self.generate_chorus(track_voc)
            return track_bac, track_voc, track_voc + track_bac, (name_bac, name_voc)

    def __len__(self):
        # Actually infinit due to the random dynamic sampling
        return int(36000 / self.frame_length)

    def readList(self, fname):
        result = []
        with open(fname, "r") as f:
            for each in f.readlines():
                each = each.strip('\n')
                result.append(each)
        return result