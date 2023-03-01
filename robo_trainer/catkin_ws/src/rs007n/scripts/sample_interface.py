#!/usr/bin/env python
import rospy
import sys
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from tf import transformations
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import MoveGroupActionResult

class Test(object):
    def __init__(self):
        rospy.init_node('sample')
        self.result = None
        self.control_pose_publisher = rospy.Publisher('/control/pose', PoseStamped, queue_size=1)
        rospy.Subscriber('/move_group/result', MoveGroupActionResult, self.result_callback)

        counter = 0
        while not rospy.is_shutdown():
            pose = PoseStamped()
            # array([6.72849573e-02, -4.75980728e-04, 3.39422800e-01,
            #       7.24784873e-01, -4.68367356e-01, 4.60060310e-01,
            #       -2.08957935e-01])
            # array([ 0.05983519, 0.0193432, 0.32863192, 0.82313186, -0.36983529,
            #       0.35868771, -0.23878635])

            pose.pose.position.x = 0.0
            pose.pose.position.y = 0.3
            pose.pose.position.z = 0.5
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 1.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 0.0
            self.control_pose_publisher.publish(pose)
            if self.result:
                print('ok')
                counter += 1

    def result_callback(self, data):
        self.result = data

if __name__ == '__main__':
    Test()
