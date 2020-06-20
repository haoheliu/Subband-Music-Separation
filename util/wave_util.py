
import wave
import numpy as np
import scipy.signal as signal
import pickle
import json


class WavObj:
    def __init__(self,fname,content):
        self.fname =fname
        self.content = content

class WaveHandler:
    def __init__(self):
        self.read_times = 0
        self.hit_times = 0
    def save_wave(self, frames, fname, bit_width=2, channels=1, sample_rate=44100):
        f = wave.open(fname, "wb")
        f.setnchannels(channels)
        f.setsampwidth(bit_width)
        f.setframerate(sample_rate)
        f.writeframes(frames.tostring())
        f.close()

    def random_chunk(self,fname,chunk_length,normalize = True):
        '''
        fname: path to wav file
        chunk_length: frame length in seconds
        '''
        f = wave.open(fname)
        params = f.getparams()
        duration = params[3] / params[2]
        sample_rate = params[2]
        sample_length = params[3]
        if(duration < chunk_length):
            raise ValueError(fname,"is not able to separate a ",chunk_length,"s chunk! With only",duration,"s")
        else:
            random_start = np.random.randint(0,sample_length-sample_rate*chunk_length)
            random_end = random_start+sample_rate*chunk_length
            f.setpos(random_start)
            raw = f.readframes(int(random_end - random_start))
            frames = np.fromstring(raw, dtype=np.short)
            if (frames.shape[0] % 2 == 1): frames = np.append(frames, 0)
            frames.shape = -1, 2
            if(normalize):
                return frames/(32768.0*(2+np.random.random()))
            else:
                return frames


    def read_wave(self, fname,
                  convert_to_mono = False,
                  portion_start = 0,
                  portion_end = 1,
                  ): # Whether you want raw bytes
        if(portion_end > 1 and portion_end < 1.1):
            portion_end = 1
        f = wave.open(fname)
        params = f.getparams()
        channel = params[0]
        if(portion_end <= 1):
            raw = f.readframes(params[3])
            frames = np.fromstring(raw, dtype=np.short)
            if(frames.shape[0] % 2 == 1):frames = np.append(frames,0)
            # Convert to mono
            frames.shape = -1, channel
            start, end = int(frames.shape[0] * portion_start), int(frames.shape[0] * portion_end)
            if(convert_to_mono):frames = frames[start:end, 0]
            else:frames = frames[start:end, :]
        else:
            f.setpos(portion_start)
            raw = f.readframes(portion_end-portion_start)
            frames = np.fromstring(raw, dtype=np.short)
            if (frames.shape[0] % 2 == 1): frames = np.append(frames, 0)
            frames.shape = -1, channel
            if(convert_to_mono):frames = frames[:,0]
            else:frames = frames[:,:]
        return frames

    def get_channels_sampwidth_and_sample_rate(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return (params[0],params[1],params[2]) == (2,2,44100),(params[0],params[1],params[2])

    def get_channels(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[0]

    def get_sample_rate(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[2]

    def get_duration(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[3]/params[2]

    def get_framesLength(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[3]

    def restore_wave(self,zxx):
        _,w = signal.istft(zxx)
        return w








