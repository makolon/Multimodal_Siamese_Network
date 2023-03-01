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
from std_msgs.msg import Header, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from moveit_commander import MoveGroupCommander
from oscilloscope import GetWave
import cv2
from PIL import Image as Img
from cv_bridge import CvBridge

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
        self.rate = rospy.Rate(3)
        self.called_count = 0

        self.group = MoveGroupCommander(self.group_name)
        self.get_wave = GetWave(self.args)

        self.start_pose = []
        self.processed_img = []

        # publisher
        self.img_publisher = rospy.Publisher("/vibration_image", Image, queue_size=10)

        # subscriber
        rospy.Subscriber("/stroke_position", Float32MultiArray, self.stroke_position_cb)

        count = 0

        offset_x = 0.0
        offset_z = 0.1420
        offset_y = 0.0950

        while not rospy.is_shutdown():
            # initialize
            if count == 0:
                rospy.loginfo("initialize")
                self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
                self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
                self.group.set_joint_value_target([0, 0, -np.pi/2, 0, -np.pi/2, 0])
                self.group.go()
                rospy.loginfo("initialized!")
                init = raw_input("initialized?: ")
                if init == "OK":
                    pass
                else:
                    continue

            if len(self.start_pose) != 0:
                pass
            else:
                continue

            count += 1

            if len(self.start_pose) < count:
                continue

            # standby cartesian space
            rospy.loginfo("reach to standby position")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            self.group.set_pose_target([0, 0.43, 0.3, -np.pi*5/6, 0., 0])
            self.group.go()
            rospy.loginfo("ready!")
            
            # x = float(self.start_pose[count-1][0])
            # y = float(self.start_pose[count-1][1] - offset_y)
            # z = float(self.start_pose[count-1][2] + offset_z)
            
            # 2022-01-21
            x1 = float(self.start_pose[count-1][0][0] - offset_x)
            y1 = float(self.start_pose[count-1][0][1] - offset_y)
            z1 = float(self.start_pose[count-1][0][2] + offset_z)

            # 2022-01-21
            x2 = float(self.start_pose[count-1][1][0] - offset_x)
            y2 = float(self.start_pose[count-1][1][1] - offset_y)
            z2 = float(self.start_pose[count-1][1][2] + offset_z)

            # rospy.loginfo("reach to touch")
            # self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.25)
            # self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            # self.group.set_pose_target([x, y, 0.3, -np.pi*5/6, 0., 0.])
            # self.group.go()
            # rospy.loginfo("ready to touch!")

            # 2022-01-21
            rospy.loginfo("reach to touch")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.25)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            self.group.set_pose_target([x1, y1, 0.3, -np.pi*5/6, 0., 0.])
            self.group.go()
            rospy.loginfo("ready to touch!")

            start = time.time()
            self.get_wave.inst.write(':RUN')
            chlist = [int(s) for s in args.chlist.split(',')]

            # rospy.loginfo("touch")
            # self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.2)
            # self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            # self.group.set_pose_target([x, y, z, -np.pi*5/6, 0., 0.])
            # self.group.go()
            # rospy.loginfo("touched!")

            # 2022-01-21
            rospy.loginfo("touch")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.2)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            self.group.set_pose_target([x1, y1, z1, -np.pi*5/6, 0., 0.])
            self.group.go()
            rospy.loginfo("touched!")

            # stroke
            # rospy.loginfo("stroke")
            # self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
            # self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.29)
            # self.group.set_pose_target([x, y-0.03, z, -np.pi*5/6, 0., 0.])
            # self.group.go()
            # rospy.loginfo("stroked!")

            # 2022-01-21
            rospy.loginfo("stroke")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.29)
            self.group.set_pose_target([x2, y2, z1, -np.pi*5/6, 0., 0.])
            self.group.go()
            rospy.loginfo("stroked!")

            flag = True
            while flag:
                if time.time() - start >= 5:
                    flag = False
                    print("5 seconds")
            self.get_wave.inst.write(':STOP')

            load_data = {}
            print("required points: ", self.args.points)
            t, v = self.get_wave.load_waveform()
            if 'time' not in load_data.keys():
                load_data['time'] = t
            load_data['CH1'] = v

            # self.plot_data(load_data)
            check = raw_input("check move: ")
            if check == "OK":
                pass
            else:
                break
            processed_data = self.preprocess(load_data, count)
            self.processed_img.append(processed_data)

            rospy.loginfo("reach to standby position")
            self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
            self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
            self.group.set_pose_target([0, 0.43, 0.3, -np.pi*5/6, 0., 0])
            self.group.go()
            rospy.loginfo("ready!")

            if len(self.start_pose) == count:
                for i in range(len(self.processed_img)):
                    self.img_publisher.publish(self.processed_img[i])
                    self.rate.sleep()
                rospy.loginfo("initialize")
                self.group.set_max_velocity_scaling_factor(self.exec_vel - 0.29)
                self.group.set_max_acceleration_scaling_factor(self.exec_acc - 0.2)
                self.group.set_joint_value_target([0, 0, -np.pi/2, 0, -np.pi/2, 0])
                self.group.go()
                rospy.loginfo("initialized!")

    # def stroke_position_cb(self, data):
    #     for i in range(len(data.data) / 3):
    #         position = []
    #         for j in range(3):
    #             position.append(data.data[j+3*i])
    #         if self.called_count == 0:
    #             self.start_pose.append(position)
    #         else:
    #             pass
    #     self.called_count += 1

    def stroke_position_cb(self, data):
        for i in range(len(data.data) / 6):
            position = []
            for k in range(2):
                pos = []
                for j in range(3):
                    pos.append(data.data[6*i+3*k+j])
                position.append(pos)
            if self.called_count == 0:
                self.start_pose.append(position)
            else:
                pass
        self.called_count += 1

    def preprocess(self, load_data, count):
        # data_time = load_data['time'][::20]
        # data_ch1 = load_data['CH1'][::20]
        data_time = load_data['time']
        data_ch1 = load_data['CH1']
        data_time = np.array(data_time)
        data_ch1 = np.array(data_ch1)
        time_shift = min(data_time)
        data_time -= time_shift
        
        # dt = 0.00001
        dt = 0.00000125

        # start_time = 80000
        start_time = 1600000

        # N = 2**16
        N = start_time
        
        data_time = data_time[start_time:start_time+N]
        data_ch1 = data_ch1[start_time:start_time+N]
        
        # fft
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
        plt.xlim(0, 50)
        plt.ylim(0, 0.02)

        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)

        file_name = str(count).zfill(3) + ".png"
        plt.savefig(file_name)
        dst = Img.open(file_name).convert("L")
        dst = dst.resize((105, 105))
        dst = np.array(dst)
        plt.clf()
        bridge = CvBridge()
        image = bridge.cv2_to_imgmsg(dst)
        return image

 
    def ajust_hight(self, object_name=None):
        hight_dict = {"aluminum_foil": 0.190, "kitchen_paper": 0.270, "cooking_paper": 0.190,
                "saran_wrap": 0.195, "noodle_red": 0.240, "noodle_blue": 0.240, 
                "wooden_cutting_board": 0.170, "plastic_cutting_board": 0.165, 
                "cloth": 0.165, "white_bowl": 0.178, "blue_bowl": 0.170, 
                "black_bowl": 0.170, "paper_bowl": 0.160, "white_dish": 0.170, "black_dish": 0.170, 
                "paper_dish": 0.165, "square_dish": 0.170, "frypan": 0.165, 
                "square_frypan": 0.165, "pot": 0.165, "white_sponge": 0.180, 
                "black_sponge": 0.180, "strainer": 0.165, "drainer": 0.170, "bowl": 0.160,
                "tea": 0.220, "calpis": 0.220, "coke": 0.220, "small_coffee": 0.215, 
                "large_coffee": 0.225, "mitten": 0.175, "brown_mug": 0.250, 
                "red_mug": 0.235, "brown_cup": 0.240, "white_mug": 0.250, "paper_cup": 0.220, 
                "metal_cup": 0.235, "straw_pot_mat": 0.175, 
                "pot_mat": 0.175, "cork_pot_mat": 0.175, "white_cup": 0.235,
                "brown_mitten": 0.175, "gray_dish": 0.170
        }
        if object_name in hight_dict.keys():
            return hight_dict[object_name]
        else:
            return 0.3

    def save(self, load_data, object_name=None):
        object_keys = {"aluminum_foil": 1, "kitchen_paper": 2, "cooking_paper": 3,
                "saran_wrap": 4, "noodle_red": 5, "noodel_blue": 6, "wooden_cutting_board": 7, 
                "plastic_cutting_board": 8, "cloth": 9,
                "white_bowl": 10, "blue_bowl": 11, "black_bowl": 12, "paper_bowl": 13,
                "white_dish": 14, "black_dish": 15, "paper_dish": 16, "square_dish": 17,
                "frypan": 18, "square_frypan": 19, "pot": 20, "white_sponge": 21,
                "black_sponge": 22, "strainer": 23, "drainer": 24, "bowl": 25,
                "tea": 26, "calpis": 27, "coke": 28, "small_coffee": 29, "large_coffee": 30,
                "mitten": 31, "brown_mug": 32, "red_mug": 33, "white_mug": 34, 
                "paper_cup": 35, "metal_cup": 36, "straw_pot_mat": 37, "pot_mat": 38,
                "cork_pot_mat": 39, "brown_cup": 40
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
