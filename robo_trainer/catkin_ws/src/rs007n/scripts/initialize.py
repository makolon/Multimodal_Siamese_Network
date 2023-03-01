#!/usr/bin/env python
import rospy
import sys
import numpy as np
import os
import argparse
import pyvisa
import csv
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
from oscilloscope import GetWave

class StrokeSample(object):
    def __init__(self, dof=4, update_rate=30):
        rospy.init_node("stroke_sample")
        self.group_name = "manipulator"
        self.enable = True
        self.dof = dof
        self.update_rate = update_rate
        self.exec_vel = 0.3

        self.group = MoveGroupCommander(self.group_name)

        while not rospy.is_shutdown():
            # standby cartesian space
            rospy.loginfo("initialize")
            self.group.set_max_velocity_scaling_factor(self.exec_vel)
            self.group.set_joint_value_target([0, 0, -np.pi/2, 0, -np.pi/2, 0])
            self.group.go()
            rospy.loginfo("initialized!")

if __name__ == "__main__":
    StrokeSample()
