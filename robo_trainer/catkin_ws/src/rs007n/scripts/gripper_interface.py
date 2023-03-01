#!/usr/bin/python
import sys
import moveit_commander
import rospy
import math
from std_msgs.msg import Float32

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
from khi_rs_gripper.msg import MoveActionGoal


class GripperInterface(object):
    def __init__(self, update_rate=10):
        self.update_rate = update_rate
        self.trigger = None
        self.group_name = 'hand'
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('gripper_interface', anonymous=True)

        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        rate = rospy.Rate(self.update_rate)

        _controller_trigger = rospy.Subscriber('/trigger/', Float32, self.trigger_callback)
        pub_gripper = rospy.Publisher('/hand/gripper_move/goal', MoveActionGoal, queue_size=1)
        pub_jt = rospy.Publisher('/hand/gripper_controller/command', JointTrajectory, queue_size=1)

        while not rospy.is_shutdown():
            if self.trigger is not None:
                gripper_goal = MoveActionGoal()
                gripper_goal.get.target_pulse = (1.0 - self.trigger.data)*0.0007
                gripper_goal.goal.pulse_speed = 1.0
                gripper_goal.header.stamp = rospy.Time.now()

                pub_gripper.publish(gripper_goal)
                print("Gripper Command: ", gripper_goal.goal.target_pulse)

            rate.sleep()

    def trigger_callback(self, data):
        self.trigger = data

if __name__ == '__main__':
    GripperInterface()
