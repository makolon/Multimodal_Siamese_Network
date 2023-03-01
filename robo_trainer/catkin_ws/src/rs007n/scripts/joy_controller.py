#!/usr/bin/env python3
import sys
import moveit_commander
import rospy
import numpy as np
import time
import copy
import tf
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Empty, String
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import InteractiveMarkerInit

def signedSquare(val):
    if val > 0:
        sign = 1
    else:
        sign = -1
    return val * val * sign

class StatusHistory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.buffer = []

    def add(self, status):
        self.buffer.append(status)
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[1 : self.max_length + 1]

    def all(self, proc):
        for status in self.buffer:
            if not proc(status):
                return False
        return True

    def latest(self):
        if len(self.buffer) > 0:
            return self.buffer[-1]
        else:
            return None

    def length(self):
        return len(self.buffer)

    def new(self, status, attr):
        if len(self.buffer) == 0:
            return getattr(status, attr)
        else:
            return getattr(status, attr) and not getattr(self.latest(), attr)

class joyController(object):
    def __init__(self):
        rospy.init_node('joy_controller')
        self.joy_pose_publisher = rospy.Publisher('/control/pose', PoseStamped, queue_size=1)
        self.joy_subscriber = rospy.Subscriber('/joy', Joy, self.joy_callback)
        self.listener = tf.TransformListener()
        self.history = StatusHistory(max_length=10)
        self.pose = None
        self.flag = False

        while not rospy.is_shutdown():
            if self.pose is not None:
                pose_goal = copy.deepcopy(self.pose)
                orn = self.pose.pose.orientation
                quat = (R.from_quat([orn.x, orn.y, orn.z, orn.w]) * R.from_euler('xyz', [0, 0, 0])).as_quat()
                pose_goal.pose.orientation.x = quat[0]
                pose_goal.pose.orientation.y = quat[1]
                pose_goal.pose.orientation.z = quat[2]
                pose_goal.pose.orientation.w = quat[3]
                self.joy_pose_publisher.publish(pose_goal)

    def joy_callback(self, msg):
        try:
            (trans, rot) = self.listener.lookupTransform('/base_link', '/link6', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        pre_pose =  PoseStamped()
        pre_pose.pose.position.x = trans[0]
        pre_pose.pose.position.y = trans[1]
        pre_pose.pose.position.z = trans[2]
        pre_pose.pose.orientation.x = rot[0]
        pre_pose.pose.orientation.y = rot[1]
        pre_pose.pose.orientation.z = rot[2]
        pre_pose.pose.orientation.w = rot[3]
        status = PS3Status(msg)
        if msg.buttons[9] == 1.0: # start button
            self.flag = True
        transformed_pose = self.computePoseFromJoy(pre_pose, status)
        self.pose = transformed_pose
        self.history.add(status)

    def computePoseFromJoy(self, pre_pose, status):
        new_pose = PoseStamped()
        # new_pose.header.frame_id = self.frame_id
        new_pose.header.stamp = rospy.Time(0.0)
        # move in local
        dist = (
            status.left_analog_y * status.left_analog_y
            + status.left_analog_x * status.left_analog_x
        )
        scale = 200.0
        x_diff = signedSquare(status.left_analog_y) / scale
        y_diff = signedSquare(status.left_analog_x) / scale
        # z
        if status.L2:
            z_diff = 0.005
        elif status.R2:
            z_diff = -0.005
        else:
            z_diff = 0.0
        if self.history.all(lambda s: s.L2) or self.history.all(lambda s: s.R2):
            z_scale = 4.0
        else:
            z_scale = 2.0
        local_move = np.array((x_diff, y_diff, z_diff * z_scale, 1.0))
        q = np.array(
            (
                pre_pose.pose.orientation.x,
                pre_pose.pose.orientation.y,
                pre_pose.pose.orientation.z,
                pre_pose.pose.orientation.w,
            )
        )
        xyz_move = np.dot(tf.transformations.quaternion_matrix(q), local_move)
        new_pose.pose.position.x = pre_pose.pose.position.x + xyz_move[0]
        new_pose.pose.position.y = pre_pose.pose.position.y + xyz_move[1]
        new_pose.pose.position.z = pre_pose.pose.position.z + xyz_move[2]
        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        DTHETA = 0.005
        if status.L1:
            if self.history.all(lambda s: s.L1):
                yaw = yaw + DTHETA * 2
            else:
                yaw = yaw + DTHETA
        elif status.R1:
            if self.history.all(lambda s: s.R1):
                yaw = yaw - DTHETA * 2
            else:
                yaw = yaw - DTHETA
        if status.up:
            if self.history.all(lambda s: s.up):
                pitch = pitch + DTHETA * 2
            else:
                yaw = yaw - DTHETA
        if status.up:
            if self.history.all(lambda s: s.up):
                pitch = pitch + DTHETA * 2
            else:
                pitch = pitch + DTHETA
        elif status.down:
            if self.history.all(lambda s: s.down):
                pitch = pitch - DTHETA * 2
            else:
                pitch = pitch - DTHETA
        if status.right:
            if self.history.all(lambda s: s.right):
                roll = roll + DTHETA * 2
            else:
                roll = roll + DTHETA
        elif status.left:
            if self.history.all(lambda s: s.left):
                roll = roll - DTHETA * 2
            else:
                roll = roll - DTHETA
        diff_q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        new_q = tf.transformations.quaternion_multiply(q, diff_q)
        new_pose.pose.orientation.x = new_q[0]
        new_pose.pose.orientation.y = new_q[1]
        new_pose.pose.orientation.z = new_q[2]
        new_pose.pose.orientation.w = new_q[3]
        return new_pose

    def stop(self):
        pass
                                      
class JoyStatus:
    def __init__(self):
        self.center = False
        self.select = False
        self.start = False
        self.L3 = False
        self.R3 = False
        self.square = False
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.triangle = False
        self.cross = False
        self.circle = False
        self.L1 = False
        self.R1 = False
        self.L2 = False
        self.R2 = False
        self.left_analog_x = 0.0
        self.left_analog_y = 0.0
        self.right_analog_x = 0.0
        self.right_analog_y = 0.0


class PS3Status(JoyStatus):
    def __init__(self, msg):
        JoyStatus.__init__(self)
        # creating from sensor_msgs/Joy
        if msg.buttons[10] == 1:
            self.center = True
        else:
            self.center = False
        if msg.buttons[8] == 1:
            self.select = True
        else:
            self.select = False
        if msg.buttons[9] == 1:
            self.start = True
        else:
            self.start = False
        if msg.buttons[11] == 1:
            self.L3 = True
        else:
            self.L3 = False
        if msg.buttons[12] == 1:
            self.R3 = True
        else:
            self.R3 = False
        if msg.buttons[3] < 0:
            self.square = True
        else:
            self.square = False
        if msg.buttons[13] < 0:
            self.up = True
        else:
            self.up = False
        if msg.buttons[14] < 0:
            self.down = True
        else:
            self.down = False
        if msg.buttons[15] < 0:
            self.left = True
        else:
            self.left = False
        if msg.buttons[16] < 0:
            self.right = True
        else:
            self.right = False
        if msg.buttons[2] < 0:
            self.triangle = True
        else:
            self.triangle = False
        if msg.buttons[0] < 0:
            self.cross = True
        else:
            self.cross = False
        if msg.buttons[1] < 0:
            self.circle = True
        else:
            self.circle = False
        if msg.buttons[4] > 0:
            self.L1 = True
        else:
            self.L1 = False
        if msg.buttons[5] > 0:
            self.R1 = True
        else:
            self.R1 = False
        if msg.buttons[6] > 0:
            self.L2 = True
        else:
            self.L2 = False
        if msg.buttons[7] > 0:
            self.R2 = True
        else:
            self.R2 = False
        self.left_analog_x = msg.axes[0]
        self.left_analog_y = msg.axes[1]
        self.right_analog_x = msg.axes[2]
        self.right_analog_y = msg.axes[3]
        self.orig_msg = msg

if __name__ == '__main__':
    joyController()