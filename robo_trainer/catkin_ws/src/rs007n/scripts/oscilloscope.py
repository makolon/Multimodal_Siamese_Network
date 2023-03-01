#!/usr/bin/env python3
import numpy as np
import time
import os
import argparse
import pyvisa
import csv

class GetWave(object):
    def __init__(self, args):
        self.inst = pyvisa.ResourceManager("/usr/lib/x86_64-linux-gnu/libvisa.so.21.0.0").open_resource(args.address,
            read_termination="\n")
        self.inst.timeout = 35000
        self.inst.chunk_size = 102400
        print(self.inst.query("*IDN?").strip())

        self.args = args
        self.inst.write(':WAV:SOUR CHAN1')
        self.inst.write(':WAV:POIN:MODE RAW')
        self.inst.write(':WAV:POIN {}'.format(self.args.points))

    def load_waveform(self):
        # self.inst.write(':WAV:SOUR CHAN{}'.format(chidx))
        # self.inst.write(':WAV:POIN:MODE RAW')
        # self.inst.write(':WAV:POIN {}'.format(points_request))

        preample = self.inst.query(':WAV:PRE?').split(',')
        points = int(preample[2])
        xinc = float(preample[4])
        xorg = float(preample[5])
        xref = float(preample[6])
        yinc = float(preample[7])
        yorg = float(preample[8])
        yref = float(preample[9])

        data_bin = self.inst.query_binary_values('WAV:DATA?', datatype='B', container=np.array)
        print('loading CH1 {}pts'.format(self.args.points))
        t = [(float(i) - xref)*xinc + xorg for i in range(self.args.points)]
        v = [(float(byte_data) - yref)*yinc + yorg for byte_data in data_bin]
        return t, v

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-a', '--address',
        help='VISA address like "TCPIP::{ipadress}::INSTR"', default="TCPIP0::192.168.0.130::inst0::INSTR")
    argparser.add_argument('-o', '--output',
        help='Output file name (default: "waveform.csv")', default='waveform.csv')
    argparser.add_argument('-p', '--points', type=int,
        help='Points of the data to be load', default=8000000)
    argparser.add_argument('-c', '--chlist',
        help='Specify channels which waveforms will be load from like "-c 1,2,3,4"', default='1')
    args = argparser.parse_args()

    device = GetWave(args)
    device.inst.write(':STOP')
    device.inst.query('*OPC?')
    # print("phase: ", device.inst.query(':MEASure:PHASe? CHAN1'))

    chlist = [int(x) for x in args.chlist.split(',')]
    load_data = {}
    for chidx in chlist:
        t, v = device.load_waveform(chidx, args.points)
        if 'time' not in load_data.keys():
            load_data['time'] = t
        load_data['CH{}'.format(chidx)] = v

    with open(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(load_data.keys())
        writer.writerows(zip(*load_data.values()))

    device.inst.close()
