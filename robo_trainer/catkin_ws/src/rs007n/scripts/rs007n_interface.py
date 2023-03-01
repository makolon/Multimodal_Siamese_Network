#!/usr/bin/env python
import sys
import moveit_commander
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import time

class rs007nInterface(object):
    def __init__(self, dof=6, update_rate=30):
        self.enable = True
        self.group_name = 'manipulator'
        self.dof = dof
        self.update_rate = update_rate
        self.pose = None

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('rs007n_interface', anonymous=True)

        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        rate = rospy.Rate(self.update_rate)

        _controller_pose = rospy.Subscriber('/control/pose', PoseStamped, self.pose_callback)
        _enable = rospy.Subscriber('/control/enable', Bool, self.enable_callback)
        
        self.pub_joint_traj = rospy.Publisher('/rs007n_arm_controller/command', JointTrajectory, queue_size=10)
        self.pub_pose = rospy.Publisher('/rs007n/pose_goal', PoseStamped, queue_size=1) # TODO: check topic name

        counter = 0

        self.enable = True

        while not rospy.is_shutdown():
            if self.enable == True:
                counter += 1
                if self.pose is not None:
                    pose_goal = copy.deepcopy(self.pose)

                    if self.dof == 3:
                        pose_goal.pose.orientation.x = -1.0
                        pose_goal.pose.orientation.y = 0.0
                        pose_goal.pose.orientation.z = 0.0
                        pose_goal.pose.orientation.w = 0.0
                    elif self.dof == 4:
                        orn = self.pose.pose.orientation
                        euler = (R.from_quat([orn.x, orn.y, orn.z, orn.w])).as_euler('zyx')
                        euler[1] = 0
                        euler[2] = np.pi
                        quat = (R.from_euler('zyx', euler)).as_quat()
                        pose_goal.pose.orientation.x = quat[0]
                        pose_goal.pose.orientation.y = quat[1]
                        pose_goal.pose.orientation.z = quat[2]
                        pose_goal.pose.orientation.w = quat[3]
                    else:
                        orn = self.pose.pose.orientation
                        quat = (R.from_quat([orn.x, orn.y, orn.z, orn.w]) * R.from_euler('xyz', [0, 0, 0])).as_quat()
                        pose_goal.pose.orientation.x = quat[0]
                        pose_goal.pose.orientation.y = quat[1]
                        pose_goal.pose.orientation.z = quat[2]
                        pose_goal.pose.orientation.w = quat[3]

                    plan, _ = self.group.compute_cartesian_path([pose_goal.pose], 0.01, 0)
                    plan.joint_trajectory.points = [plan.joint_trajectory.points[-1]]
                    plan.joint_trajectory.points[-1].time_from_start = rospy.Duration(1. / self.update_rate)

                    stamp = rospy.Time.now()
                    pose_goal.header.stamp = stamp
                    plan.joint_trajectory.header.stamp = stamp
                    
                    self.pub_pose.publish(pose_goal)
                    self.pub_joint_traj.publish(plan.joint_trajectory)

            rate.sleep()
        
    def pose_callback(self, data):
        self.pose = data
        print(self.pose)

    def enable_callback(self, data):
        if data == True:
            jt = JointTrajectory()
            jt.header.stamp = rospy.Time.now()
            jt.points = []
            self.pub_joint_traj.publish(jt)
            self.pose = self.group.getPose()
        self.enable = data

    def movehome_callback(self, data):
        self.enable = False
        pose_home = PoseStamped()
        # TODO: check pose_home
        pose_home.pose.position.x = 0.47
        pose_home.pose.position.y = 0.0
        pose_home.pose.position.z = 0.54
        pose_home.pose.orientation.x = -1.0
        pose_home.pose.orientation.y = 0.0
        pose_home.pose.orientation.z = 0.0
        pose_home.pose.orientation.w = 0.0
        plan, _ = self.group.compute_cartesian_path([pose_home.pose], 0.01, 0)
        robot = moveit_commander.RobotCommander()
        plan = self.group.retime_trajectory(
            robot.get_current_state(),
            plan,
            velocity_scaling_factor=0.1,
            acceleration_scaling_factor=0.1,
            algorithm="time_optimal_trajectory_generation"
        )
        self.group.execute(plan)

        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.points = []
        self.pub_joint_traj.publish(jt)
        self.pose = pose_home
        
        time.sleep(0.5)

        self.enable = True
    
    def emergency(self):
        return False

if __name__ == '__main__':
    rs007nInterface()
