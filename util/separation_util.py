import time
import os
import torch
import numpy as np
import sys
import musdb
import logging
sys.path.append("..")
from models import dedicated
from util import wave_util
from util.data_util import write_json,save_pickle,load_json
from util.subband.subband_util import before_forward_f,after_forward_f
from evaluate.museval.evaluate_track import eval_mus_track

class SeparationUtil:
    def __init__(self, model,
                 device = torch.device('cpu'),
                 project_root = "",
                 trail_name = "temp",
                 MUSDB_PATH="",
                 split_band = 1,
                 sample_rate = 44100,
                 ):
        '''
        Args:
            model: model object, defined in model.dedicated.dedicated_model
            device: torch.device, cpu or cuda:n
            project_root: str, root path to this project
            trail_name: str, name alias of this experiment/ model
            MUSDB_PATH: str, root path to MUSDB18 dataset
            split_band: int, how many subband to split
            sample_rate: int, default 44100
        '''
        self.project_root = project_root
        self.MUSDB_PATH = MUSDB_PATH
        self.sample_rate = sample_rate
        self.split_band = split_band
        self.wh = wave_util.WaveHandler()
        self.device = device
        self.model_name = trail_name
        self.model = model
        self.start_point = self.model.cnt
        # end else
        self.start = []
        self.end = []
        self.realend = []
        if(self.split_band == 1):
            self.tiny_segment = 2
            for each in np.linspace(0, 995, 200):
                self.start.append(each / 1000)
                self.end.append((each + 5 + self.tiny_segment) / 1000)
                self.realend.append((each + 5) / 1000)
        else:
            self.tiny_segment = 5
            for each in np.linspace(0, 950, 20):
                self.start.append(each / 1000)
                self.end.append((each + 50 + self.tiny_segment) / 1000)
                self.realend.append((each + 50) / 1000)

    def seg(self,audio,portion_start,portion_end):
        length = audio.shape[0]
        return audio[int(length*portion_start):int(length*portion_end),...]

    def pre_pro(self,tensor: torch.Tensor):
        if(len(tensor.size()) <= 2):
            tensor = tensor.unsqueeze(0)
        assert len(tensor.size()) == 3
        if('cuda' in str(self.device)):return tensor.permute(0, 2, 1).float().cuda(self.device)
        else: return tensor.permute(0, 2, 1).float()

    def post_pro(self,arr:np.array,real_length):
        if(len(arr.shape)>2):
            arr = arr[0,...]
        arr = np.transpose(arr,(1,0))
        return arr[:real_length,...]

    def validate(self,previous_loss):
        '''
        Args:
            previous_loss: tuple, previously the best loss value

        Returns:
            tuple, Validation loss value
        '''
        self.mus = musdb.DB(self.MUSDB_PATH, is_wav=True, subsets='train',split='valid')
        loss = torch.nn.L1Loss()
        conf = load_json(self.project_root+"config/json/"+self.model_name+".json")
        decrease_ratio = conf['decrease_ratio']
        bac_loss = []
        voc_loss = []
        t_start = time.time()
        with torch.no_grad():
            for track in self.mus:
                print(track.name)
                # if("Alexander Ross - Goodbye Bolero" in track.name): # todo this song is broken on my server
                #     continue
                bac = track.targets['accompaniment'].audio
                voc = track.targets['vocals'].audio
                for i in range(len(self.start)):
                    portion_start, portion_end, real_end = self.start[i], self.end[i], self.realend[i]
                    reference_bac = self.seg(bac, portion_start, real_end)
                    reference_voc = self.seg(voc, portion_start, real_end)
                    input_bac = self.pre_pro(torch.Tensor(reference_bac))
                    input_voc = self.pre_pro(torch.Tensor(reference_voc))
                    input_f_background, input_f_vocals = before_forward_f(input_bac, input_voc,
                                                                          subband_num=self.split_band,
                                                                          device=self.device,
                                                                          sample_rate=self.sample_rate,
                                                                          normalize=False)
                    input_f = (input_f_vocals + input_f_background)
                    self.model.eval()
                    out_bac = input_f * self.model(0,input_f)
                    out_voc = input_f * self.model(1,input_f)
                    self.model.train()
                    bac_loss.append(float(loss(input_f_background,out_bac)))
                    voc_loss.append(float(loss(input_f_vocals,out_voc)))
        t_end = time.time()
        ret = (np.average(bac_loss),np.average(voc_loss))
        print("decrease-rate-threshold:",decrease_ratio)
        print("Validation time usage:",t_end-t_start,"s")
        print("Result:   ","bac-",ret[0],"voc-",ret[1])
        print("Previous: ","bac-",previous_loss[0],"voc-",previous_loss[1])
        if(previous_loss[0] is None):
            return ret
        if(ret[0]/previous_loss[0] < decrease_ratio or ret[1]/previous_loss[1] < decrease_ratio):
            try:
                print("Save model")
                torch.save(self.model.state_dict(), self.project_root+"saved_models/" + self.model_name + "/model" + str(self.model.cnt) + ".pth")
                self.evaluate(save_wav=True,save_json=True)
                self.split_listener()
                return ret
            except Exception as e:
                logging.exception(e)
                return ret
        else:
            return previous_loss


    def evaluate(self, save_wav=True,save_json=True):
        '''
        Do evaluation on MUSDB18 test set
        Args:
            save_wav: boolean,
            save_json: boolean, save result json
        '''
        def __fm(num):
            return format(num,".2f")

        def __get_aproperate_keys():
            keys = []
            for each in list(res.keys()):
                if ("ALL" not in each):
                    keys.append(each)
            return keys

        def __get_key_average(key,keys):
            util_list = [res[each][key] for each in keys]
            return np.mean(util_list)  # sum(util_list) / (len(util_list) - 1)

        def __get_key_median(key,keys):
            util_list = [res[each][key] for each in keys]
            return np.median(util_list)

        def __get_key_std(key,keys):
            util_list = [res[each][key] for each in keys]
            return np.std(util_list)

        def __roc_val(item, key: list, value: list):
            for each in zip(key, value):
                res[item][each[0]] = each[1]

        def __cal_avg_val(keys: list):
            proper_keys = __get_aproperate_keys()
            for each in keys:
                res["ALL_median"][each] = 0
                res["ALL_mean"][each] = 0
                res["ALL_std"][each] = 0
                res["ALL_median"][each] = __get_key_median(each,proper_keys)
                res["ALL_mean"][each] = __get_key_average(each,proper_keys)
                res["ALL_std"][each] = __get_key_std(each,proper_keys)
                print(each,":")
                print( __fm(res["ALL_median"][each]),",", __fm(res["ALL_mean"][each]),",",__fm(res["ALL_std"][each]))

        self.mus = musdb.DB(self.MUSDB_PATH, is_wav=True, subsets='test')
        json_file_alias = self.project_root + "outputs/musdb_test/" + self.model_name + str(
            self.start_point) + "/result_" + self.model_name + str(self.start_point) + ".json"
        bac_keys = ["mus_sdr_bac", "mus_isr_bac", "mus_sir_bac", "mus_sar_bac"]
        voc_keys = ["mus_sdr_voc", "mus_isr_voc", "mus_sir_voc", "mus_sar_voc"]
        save_pth = self.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point)
        # if(os.path.exists(save_pth)):
        #     print("Already exist: ", save_pth)
        #     return
        if (os.path.exists(json_file_alias+"@")): # todo here we just do not want this program to find these json file
            res = load_json(json_file_alias)
            # print("Find:",res)
            res["ALL_median"] = {}
            res["ALL_mean"] = {}
            res["ALL_std"] = {}
        else:
            res = {}
            res["ALL_median"] = {}
            res["ALL_mean"] = {}
            res["ALL_std"] = {}
            dir_pth = self.test_pth
            pth = os.listdir(dir_pth)
            pth.sort()
            for cnt,track in enumerate(self.mus):
                # print("evaluating: ", track.name)
                res[track.name] = {}
                try:
                    print("......................")
                    background, vocal, origin_background, origin_vocal = self.split(track,
                                          save=save_wav,
                                          save_path= save_pth + "/",
                                          fname = track.name,
                                          )
                    eval_targets = ['vocals', 'accompaniment']
                    origin, estimate = {}, {}
                    origin[eval_targets[0]], origin[eval_targets[1]] = origin_vocal, origin_background
                    estimate[eval_targets[0]], estimate[eval_targets[1]] = vocal, background
                    data = eval_mus_track(origin, estimate,
                                          output_dir=save_pth,
                                          track_name=track.name)
                    print(data)
                    museval_res = data.get_result()
                    bac_values = [museval_res['accompaniment']['SDR'], museval_res['accompaniment']['ISR'],
                                  museval_res['accompaniment']['SIR'], museval_res['accompaniment']['SAR']]
                    voc_values = [museval_res['vocals']['SDR'], museval_res['vocals']['ISR'],
                                  museval_res['vocals']['SIR'], museval_res['vocals']['SAR']]
                    __roc_val(track.name, bac_keys, bac_values)
                    __roc_val(track.name, voc_keys, voc_values)

                except Exception as e:
                    print("ERROR: splitting error...")
                    logging.exception(e)

        print("Result:")
        print("Median,", "Mean,", "Std")
        __cal_avg_val(bac_keys)
        __cal_avg_val(voc_keys)

        if (save_json == True):
            if (not os.path.exists(self.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point))):
                os.mkdir(self.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point))
            write_json(res, self.project_root + "outputs/musdb_test/" + self.model_name + str(
                self.start_point) + "/result_" + self.model_name + str(self.start_point) + ".json")

    def unify_energy(self,audio):
        upper = 0.4
        val_max = np.max(audio)
        if(val_max<0.001):return audio
        else:return audio*(upper/val_max)

    def split(self,
              track,
              save=True,
              fname="temp",
              save_path="",
              ):
        '''
        Split a single .wav file
        Args:
            track: str (path to wav) or musdb.audio_classes.MultiTrack
            save: save result or not
            save_path: path to save result
            fname: name of this track
        '''
        if (save_path[-1] != '/'):
            raise ValueError("Error: path should end with /")
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)
        background = None
        vocal = None
        if(type(track) == type("")):
            origin_background = self.wh.read_wave(track)/2
            origin_vocal = self.wh.read_wave(track)/2
            if(np.max(origin_vocal) > 2):
                origin_vocal,origin_background = self.unify_energy(origin_vocal/32768),self.unify_energy(origin_background/32768)
            else:
                origin_vocal,origin_background = self.unify_energy(origin_vocal),self.unify_energy(origin_background)
        else:
            origin_background = track.targets['accompaniment'].audio
            origin_vocal = track.targets['vocals'].audio
        start = time.time()
        if(not os.path.exists(save_path + "background_"+fname+".wav") or not os.path.exists(save_path + "vocal_"+fname+".wav")):
            print("Not found: ",save_path + "background_"+fname+".wav",save_path + "vocal_"+fname+".wav")
            with torch.no_grad():
                for i in range(len(self.start)):
                    portion_start, portion_end, real_end = self.start[i], self.end[i], self.realend[i]
                    reference_background = self.seg(origin_background,portion_start,real_end)
                    input_background = self.seg(origin_background,portion_start,portion_end)
                    input_vocals = self.seg(origin_vocal,portion_start,portion_end)
                    # Construct Spectrom of Song
                    input_background = self.pre_pro(torch.Tensor(input_background))
                    input_vocals = self.pre_pro(torch.Tensor(input_vocals))
                    input_f_background,input_f_vocals = before_forward_f(input_background,input_vocals,
                                                                         subband_num=self.split_band,
                                                                         device=self.device,
                                                                         sample_rate=self.sample_rate,
                                                                        normalize = False)
                    input_f = (input_f_vocals + input_f_background)
                    # Forward and mask
                    self.model.eval()
                    for ch in range(self.model.sources):
                        mask = self.model(ch, input_f)
                        data = mask * input_f * scale
                        if (ch == 0):
                            construct_background = after_forward_f(data,
                                                                   subband_num=self.split_band,
                                                                   device=self.device,
                                                                   sample_rate=self.sample_rate,
                                                                   normalized=False)
                            construct_background = construct_background.cpu().detach().numpy()
                        else:
                            construct_vocal = after_forward_f(data,
                                                              subband_num=self.split_band,
                                                              device=self.device,
                                                              sample_rate=self.sample_rate,
                                                              normalized=False)
                            construct_vocal = construct_vocal.cpu().detach().numpy()
                    self.model.train()
                    # if(test_mode):break
                    real_length = reference_background.shape[0]
                    construct_background = self.post_pro(construct_background,real_length)
                    construct_vocal = self.post_pro(construct_vocal,real_length)
                    if(background is None and vocal is None):
                        background = construct_background
                        vocal = construct_vocal
                    else:
                        background = np.vstack((background, construct_background))
                        vocal = np.vstack((vocal, construct_vocal))
            if (save == True):
                if (not os.path.exists(save_path)):
                    print("Creat path", save_path)
                self.wh.save_wave((background*32768).astype(np.int16), save_path + "background_" + fname + ".wav",
                                  channels=2)
                self.wh.save_wave((vocal*32768).astype(np.int16), save_path + "vocal_" + fname + ".wav", channels=2)
                print("Split work finish!")
        else:
            background = self.wh.read_wave(save_path + "background_"+fname+".wav")/32768.0
            vocal = self.wh.read_wave(save_path + "vocal_"+fname+".wav")/32768.0

        end = time.time()
        print('time cost', end - start, 's')
        return background, vocal, origin_background, origin_vocal

    def split_listener(self, fname=None):
        '''
        Split songs in "evaluate/listener_todo/" and save result in "outputs/listener/"
        Args:
            fname: optional, str, if not specified all songs in "evaluate/listener_todo/" will be splitted
        '''
        pth = self.project_root + "evaluate/listener_todo/"
        output_path = self.project_root + "outputs/listener/" + self.model_name + str(self.start_point) + "/"
        if (not os.path.exists(output_path)):
            os.mkdir(output_path)
        else:
            print("Already exist: ",output_path)
        for each in os.listdir(pth):
            if (each.split('.')[-1] != 'wav'):
                continue
            if (fname is not None and fname != each):
                continue
            file = each.split('.')[-2]
            self.split(pth + each,
                       save=True,
                       save_path=output_path,
                       fname=file)

if __name__ == "__main__":
    from config.mainConfig import Config
    from models.dedicated import dedicated_model

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

    model.load_state_dict(torch.load(os.path.join(Config.load_model_path,"model"+str(Config.start_point)+".pth")))
    su = SeparationUtil(model=model,
                        device=Config.device,
                        MUSDB_PATH=Config.MUSDB18_PATH,
                        split_band=Config.subband,
                        sample_rate=Config.sample_rate,
                        project_root=Config.project_root,
                        trail_name="Demo code v1")

    su.split_listener()
