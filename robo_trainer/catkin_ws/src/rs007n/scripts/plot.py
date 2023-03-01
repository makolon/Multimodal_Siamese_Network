# coding: utf-8
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy
from scipy import signal

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y

data = pd.read_csv('~/rs007n_docker/catkin_ws/src/rs007n_launch/src/waveform.csv', names=['CH1', 'time'], encoding='UTF8', header=0, index_col=None)

time_shift = min(data['time'])
data['time'] -= time_shift

N = len(data["CH1"])
dt = data["time"][1] - data["time"][0]
freq = 1/dt

plt.plot(data['time'], data['CH1'])
plt.show()

# lowpass
fp = 6000
fs = 3000
gpass = 3
gstop = 40
samplerate = 1/dt

data_raw = np.array([data['time'], data['CH1']])
data_filt = lowpass(data_raw, samplerate, fp, fs, gpass, gstop)

plt.plot(data['time'], data_filt[1])
plt.show()

x = data["CH1"][2**10:2**20]
x_filt = data_filt[1][2**10:2**20]

# fft raw
N = len(x)
fft_data = np.abs(np.fft.fft(x))
fft_abs_amp = fft_data / N * 2
fft_abs_amp[0] = fft_abs_amp[0] / 2
fq = np.linspace(0, 1000, N)

plt.plot(fq, fft_abs_amp)
plt.show()

# fft lowpass
fft_data_low = np.abs(np.fft.fft(x_filt))
fft_abs_amp = fft_data_low / N * 2
fft_abs_amp[0] = fft_abs_amp[0] / 2
fq = np.linspace(0, 1000, N)

plt.plot(fq, fft_abs_amp)
plt.show()


# freqList = np.fft.fftfreq(1000, d=1/10000)
# plt.plot(freqList, fft_data)
