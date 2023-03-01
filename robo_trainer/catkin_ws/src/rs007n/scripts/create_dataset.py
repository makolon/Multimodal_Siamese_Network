from numpy.typing import _128Bit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy
from scipy import signal
import librosa
import os

class CreateDataset():
    def __init__(self):
        self.object_keys = {"aluminum_foil": 1, "kitchen_paper": 2, "cooking_paper": 3,
            "saran_wrap": 4, "noodle_red": 5, "noodle_blue": 6, "wooden_cutting_board": 7, 
            "plastic_cutting_board": 8, "cloth": 9,
            "white_bowl": 10, "blue_bowl": 11, "black_bowl": 12, "paper_bowl": 13,
            "white_dish": 14, "black_dish": 15, "paper_dish": 16, "square_dish": 17,
            "frypan": 18, "square_frypan": 19, "pot": 20, "white_sponge": 21,
            "black_sponge": 22, "strainer": 23, "drainer": 24, "bowl": 25,
            "tea": 26, "calpis": 27, "coke": 28, "small_coffee": 29, "large_coffee": 30,
            "mitten": 31, "brown_mug": 32, "red_mug": 33, "white_mug": 34, 
            "paper_cup": 35, "metal_cup": 36, "straw_pot_mat": 37, "pot_mat": 38,
            "cork_pot_mat": 39, "brown_cup": 40, "white_cup": 41, "brown_mitten": 42,
            "gray_dish": 43
        }
        self.data_path = {}
        self.save_cropped_data = True
        self.log = True
        self.output_path = "/home/designlab/rs007n_docker/catkin_ws/src/rs007n_launch/src/dataset"
        path = "/home/designlab/rs007n_docker/catkin_ws/src/rs007n_launch/src/wav_data/"
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for subdirectory in folders:
            folders2 = [f for f in os.listdir(os.path.join(
                path, subdirectory)) if not f[0] == '.']
            tmp_folder = []
            for file in folders2: 
                idx = int(file[:2])
                tmp_folder.append(
                    (os.path.join(path, subdirectory, file))
                )
            self.data_path[idx] = tmp_folder

    def lowpass_filter(data, samplerate, fp, fs, gpass, gstop):
        fn = samplerate / 2
        wp = fp / fn
        ws = fs / fn
        N, Wn = signal.buttord(wp, ws, gpass, gstop)
        if N == 0:
            N = 1
        b, a = signal.butter(N, Wn, "low")
        data_time, data_ch1 = signal.filtfilt(b, a, data)
        return data_time, data_ch1

    def load_data(self, object_name):
        data_idx = self.object_keys[object_name]
        datasets = self.data_path[data_idx]
        for idx, data_path in enumerate(datasets):
            data = pd.read_csv(data_path, names=['CH1', 'time'], encoding='UTF8', header=0, index_col=None)

            time_shift = min(data['time'])
            data['time'] -= time_shift
            dt = data['time'][1] - data['time'][0]
            freq = 1 / dt
            samplerate = len(data['CH1']) / 5

            # fp = 100
            # fs = 10
            # gpass = 3
            # gstop = 40

            # fn = samplerate / 2
            # wp = fp / fn
            # ws = fs / fn
            # N, Wn = signal.buttord(wp, ws, gpass, gstop)
            # if N == 0:
            #     N = 1
            # data = np.array([data['time'], data['CH1']])
            # b, a = signal.butter(N, Wn, "low")
            # data_time, data_ch1 = signal.filtfilt(b, a, data)

            data_time = data["time"]
            data_ch1 = data["CH1"]

            if self.save_cropped_data:
                print(object_name)
                # data_time, data_ch1 = self.save_raw(data, idx, object_name)
                # cropped_data_ch1, cropped_data_time = self.crop_from_touch(data_time, data_ch1, idx, object_name)
                cropped_data_ch1, cropped_data_time = self.thin_out(data_time, data_ch1, idx, object_name)
                # cropped_data_ch1, cropped_data_time = self.preprocess_crop(data, idx, object_name)
                # self.preprocess_fft(cropped_data_ch1, cropped_data_time, dt, idx, object_name, N=2**18)

    def thin_out(self, data_time, data_ch1, idx, object_name):
        # data_time = data_time[::20]
        # data_ch1 = data_ch1[::20]
        plt.plot(data_time, data_ch1, color="b")
        plt.xlim(min(data_time), max(data_time))
        plt.ylim(-0.25, 0.25)

        file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
        output_path = os.path.join(self.output_path, object_name)
        output_path = os.path.join(output_path, "raw")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, file_name)
        plt.savefig(output_file)
        plt.close()

        # dt = 0.00001
        dt = 0.00000125
        # start_time = 44000
        # start_time = 80000
        start_time = 1600000

        # from touch
        data_time = data_time[start_time:start_time*2]
        data_ch1 = data_ch1[start_time:start_time*2]
        plt.plot(data_time, data_ch1, color="b")
        plt.xlim(min(data_time), max(data_time))
        plt.ylim(-0.25, 0.25)

        file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
        output_path = os.path.join(self.output_path, object_name)
        output_path = os.path.join(output_path, "crop")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, file_name)
        plt.savefig(output_file)
        plt.close()

        self.preprocess_fft(data_time, data_ch1, dt, idx, object_name, N=1600000)
        return data_time, data_ch1

    def preprocess_fft(self, data_time, data_ch1, dt, idx, object_name, N=2**21):
        fft_data = np.abs(np.fft.fft(data_ch1))
        fft_abs_amp = fft_data / N * 2
        fft_abs_amp[0] = 0
        fq = np.linspace(0, 1.0/dt, N)

        fn = 1/dt/2
        fft_abs_amp[(fq>fn)] = 0
        fft_abs_amp[(fq<=0.03)] = 0

        window = np.ones(5)/5
        F = np.convolve(fft_abs_amp, window, mode="same")

        plt.plot(fq, F, color="b", linewidth=3)
        # plt.plot(fq, fft_abs_amp, color="b", linewidth=3)
        plt.xlim(0, 50)
        plt.ylim(0, 0.02)

        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)

        file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
        output_path = os.path.join(self.output_path, object_name)
        output_path = os.path.join(output_path, "fft")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, file_name)
        plt.savefig(output_file)
        plt.close()

        if self.log:
            log_fft = 10*np.log10(fft_abs_amp)
            plt.plot(fq, log_fft, color="b")
            plt.xlim(-0.5, 1000)
            plt.ylim(-100, 0)
            file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
            output_path = os.path.join(self.output_path, object_name)
            output_path = os.path.join(output_path, "log_fft")
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, file_name)
            plt.savefig(output_file)
            plt.close()

    def crop_from_touch(self, data, idx, object_name, N=2**21):
        max_idxs = np.argmax(data['CH1'])
        data_time = data['time'][max_idxs+120000:max_idxs+N]
        data_time -= min(data_time)
        data_ch1 = data['CH1'][max_idxs+120000:max_idxs+N]
        plt.plot(data_time, data_ch1, color="b")
        plt.xticks([0, 1, 2, 3])
        plt.xlim(min(data_time), max(data_time))
        plt.ylim(-0.25, 0.25)

        file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
        output_path = os.path.join(self.output_path, object_name)
        output_path = os.path.join(output_path, "crop_from_touch")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, file_name)
        plt.savefig(output_file)
        plt.close()
        return data_time, data_ch1

    def preprocess_crop(self, data, idx, object_name, start_time=720000, N=2**21):
        data_time = data['time'][start_time:start_time+N]
        data_ch1 = data['CH1'][start_time:start_time+N]
        plt.plot(data_time, data_ch1, color="b")
        plt.xlim(min(data_time), max(data_time))
        plt.ylim(-0.25, 0.25)

        file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
        output_path = os.path.join(self.output_path, object_name)
        output_path = os.path.join(output_path, "crop")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, file_name)
        plt.savefig(output_file)
        plt.close()
        return data_time, data_ch1

    def save_raw(self, data, idx, object_name):
        data_time = data['time']
        data_ch1 = data['CH1']
        plt.plot(data['time'], data['CH1'], color="b")
        plt.xlim(min(data_time), max(data_time))
        plt.ylim(-0.25, 0.25)

        file_name = str(self.object_keys[object_name]).zfill(4) + "-" + \
            str(idx).zfill(2) + ".png"
        output_path = os.path.join(self.output_path, object_name)
        output_path = os.path.join(output_path, "raw")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, file_name)
        plt.savefig(output_file)
        plt.close()
        return data_time, data_ch1


if __name__ == "__main__":
    object_keys = {"aluminum_foil": 1, "kitchen_paper": 2, "cooking_paper": 3,
        "saran_wrap": 4, "noodle_red": 5, "noodle_blue": 6, "wooden_cutting_board": 7, 
        "plastic_cutting_board": 8, "cloth": 9,
        "white_bowl": 10, "blue_bowl": 11, "black_bowl": 12, "paper_bowl": 13,
        "white_dish": 14, "black_dish": 15, "paper_dish": 16, "square_dish": 17,
        "frypan": 18, "square_frypan": 19, "pot": 20, "white_sponge": 21,
        "black_sponge": 22, "strainer": 23, "drainer": 24, "bowl": 25,
        "tea": 26, "calpis": 27, "coke": 28, "small_coffee": 29, "large_coffee": 30,
        "mitten": 31, "brown_mug": 32, "red_mug": 33, "white_mug": 34, 
        "paper_cup": 35, "metal_cup": 36, "straw_pot_mat": 37, "pot_mat": 38,
        "cork_pot_mat": 39, "brown_cup": 40, "white_cup": 41, "brown_mitten": 42,
        "gray_dish": 43
    }
    create_data = CreateDataset()
    for obj in object_keys.keys():
        create_data.load_data(obj)