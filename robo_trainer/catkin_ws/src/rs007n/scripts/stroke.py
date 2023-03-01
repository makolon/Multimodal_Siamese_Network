#!/usr/bin/env python
import rospy
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pyvisa
import csv
import time
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
from oscilloscope import GetWave

class StrokeSample(object):
    def __init__(self, args, dof=4, update_rate=30):
        rospy.init_node("stroke_sample")
        self.group_name = "manipulator"
        self.enable = True
        self.dof = dof
        self.update_rate = update_rate
        self.exec_vel = 0.3
        self.exec_acc = 0.3
        self.args = args

        self.group = MoveGroupCommander(self.group_name)

        # self.inst = pyvisa.ResourceManager("/usr/lib/x86_64-linux-gnu/libvisa.so.21.0.0").open_resource(self.args.address,
        #     read_termination="\n")
        # self.inst.timeout = 30000
        # self.inst.chunk_size = 102400
        # print(self.inst.query("*IDN?").strip())
        self.get_wave = GetWave(self.args)
        
        object_name = None
        object_name = raw_input('input object name: ')
        count = 0
        check = "OK"
        while not rospy.is_shutdown():
            # standby joint space
            # rospy.loginfo("reach to standby position")
            # self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.1)
            # self.group.set_joint_value_target([0, 0.6896120417171814, 
            #     -1.63242821788295, 0, -np.pi/24, 0.0006590793011485461]) # joint5: 0.8189788888526336
            # self.group.go()
            # rospy.loginfo("ready!")

            # standby cartesian space
            rospy.loginfo("reach to standby position")
            self.group.set_max_velocity_scaling_factor(self.exec_vel-0.29)
            self.group.set_pose_target([0, 0.43, 0.3, -np.pi*5/6, 0., 0])
            self.group.go()
            rospy.loginfo("ready!")

            # stroke joint space
            # rospy.loginfo("stroke")
            # self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.1)
            # self.group.set_joint_value_target([0, np.pi/3, 
            #     -1.63242821788295, 0, 0.1, 0.0006590793011485461]) 
            # 0.8189788888526336
            # self.group.go()
            # rospy.loginfo("stroked!")
            
            # touch to the object
            # object_name = raw_input('input object name: ')
            z = self.ajust_hight(object_name)
    
            start = time.time()
            self.get_wave.inst.write(':RUN')
            chlist = [int(x) for x in args.chlist.split(',')]

            rospy.loginfo("touch")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.2)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            # self.group.set_pose_target([0, 0.5, 0.1, 3.14159265, 0, 0.5])
            self.group.set_pose_target([0, 0.43, z, -np.pi*5/6, 0., 0])
            self.group.go()
            rospy.loginfo("touched!")

            # stroke
            rospy.loginfo("stroke")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.29)
            # self.group.set_pose_target([0, 0.5, 0.1, 3.14159265, 0, 0.5])
            self.group.set_pose_target([0, 0.40, z, -np.pi*5/6, 0., 0])
            self.group.go()
            rospy.loginfo("stroked!")

            flag = True
            while flag:
                if time.time() - start == 5:
                    flag = False
                    print("5 sec")
            self.get_wave.inst.write(':STOP')

            rospy.loginfo("return to standby position")
            self.group.set_max_velocity_scaling_factor(self.exec_vel)
            self.group.set_pose_target([0, 0.43, 0.3, -np.pi*5/6, 0., 0])
            self.group.go()
            rospy.loginfo("returned!")

            # self.get_wave.inst.write(':STOP')

            load_data = {}
            print("required points: ", self.args.points)
            t, v = self.get_wave.load_waveform()
            if 'time' not in load_data.keys():
                load_data['time'] = t
            load_data['CH1'] = v

            # self.plot_data(load_data)
            if count == 0:
                check = raw_input("check move: ")
            if check == "OK":
                pass
            else:
                break
            self.save(load_data, object_name)
            count += 1
    
    def ajust_hight(self, object_name):
        hight_dict = {"aluminum_foil": 0.190, "kitchen_paper": 0.271, "cooking_paper": 0.190,
                "saran_wrap": 0.195, "noodle_red": 0.241, "noodle_blue": 0.241, 
                "wooden_cutting_board": 0.169, "plastic_cutting_board": 0.163, 
                "cloth": 0.167, "white_bowl": 0.178, "blue_bowl": 0.170, 
                "black_bowl": 0.169, "paper_bowl": 0.162, "white_dish": 0.169, "black_dish": 0.169, 
                "paper_dish": 0.165, "square_dish": 0.171, "frypan": 0.165, 
                "square_frypan": 0.165, "pot": 0.165, "white_sponge": 0.180, 
                "black_sponge": 0.180, "strainer": 0.165, "drainer": 0.168, "bowl": 0.162,
                "tea": 0.220, "calpis": 0.220, "coke": 0.220, "small_coffee": 0.213, 
                "large_coffee": 0.226, "mitten": 0.175, "brown_mug": 0.248, 
                "red_mug": 0.236, "brown_cup": 0.238, "white_mug": 0.25, "paper_cup": 0.220, 
                "metal_cup": 0.233, "straw_pot_mat": 0.173, 
                "pot_mat": 0.173, "cork_pot_mat": 0.173, "white_cup": 0.233,
                "brown_mitten": 0.175, "gray_dish": 0.170
        }
        if object_name in hight_dict.keys():
            return hight_dict[object_name]
        else:
            return 0.3

    def save(self, load_data, object_name):
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
        if object_name in object_keys.keys():
            pass
        else:
            return
        output_path = os.path.join(self.args.output, object_name)
        print("output_path: ", output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        num_files = sum(os.path.isfile(os.path.join(output_path, name)) for name in os.listdir(output_path))
        output_file = os.path.join(output_path, str(object_keys[object_name]).zfill(2) + "-" + str(num_files).zfill(4) + ".csv") 
        with open(output_file, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(load_data.keys())
            writer.writerows(zip(*load_data.values()))

    def plot_data(self, load_data):
        plt.plot(load_data['time'], load_data['CH1'])
        print(len(load_data["time"]))
        print(len(load_data["CH1"]))
        plt.xlim(0, 5)
        plt.ylim(-0.1, 0.1)
        plt.show()

    def load_waveform(self, chidx, points_request):
        self.get_wave.inst.write(':WAV:SOUR CHAN{}'.format(chidx))
        self.get_wave.inst.write(':WAV:POIN:MODE RAW')
        self.get_wave.inst.write(':WAV:POIN {}'.format(points_request))

        preample = self.get_wave.inst.query(':WAV:PRE?').split(',')
        points = int(preample[2])
        xinc = float(preample[4])
        xorg = float(preample[5])
        xref = float(preample[6])
        yinc = float(preample[7])
        yorg = float(preample[8])
        yref = float(preample[9])

        data_bin = self.get_wave.inst.query_binary_values('WAV:DATA?', datatype='B', container=bytes)
        print('loading CH{} {}pts'.format(chidx, points))
        t = [(float(i) - xref)*xinc + xorg for i in range(points)]
        v = [(float(byte_data) - yref)*yinc + yorg for byte_data in data_bin if byte_data != "(" and byte_data != "," and byte_data != ")" and byte_data != " "]
        return t, v
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-a', '--address',
        help='VISA address like "TCPIP::{ipadress}::INSTR"', default="TCPIP0::192.168.0.130::inst0::INSTR")
    argparser.add_argument('-o', '--output',
        help='Output file name (default: "waveform.csv")', default='wav_data')
    argparser.add_argument('-p', '--points', type=int,
        help='Points of the data to be load', default=4000000)
    argparser.add_argument('-c', '--chlist',
        help='Specify channels which waveforms will be load from like "-c 1,2,3,4"', default='1')
    args = argparser.parse_args()
    StrokeSample(args)
