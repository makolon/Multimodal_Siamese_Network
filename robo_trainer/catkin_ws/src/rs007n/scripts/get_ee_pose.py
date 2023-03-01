#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Float32

class GetEEPose:
    def __init__(self):
        rospy.init_node('get_ee_pose', anonymous=True)
        self.transforms = None

        # subscriber
        rospy.Subscriber('/tf', TFMessage, self._transforms_cb)

        # transforms
        self.tm = TransformManager()
        self.tm.add_transform("link0", "robot", pt.transform_from_pq([0, 0, 0, 0, 0, 0, 0]))
    
    def ee_pose(self):
        ee_pose = self._transform_ee_pose(self.transforms)
        return ee_pose

    def _transforms_cb(self, data):
        self.transforms = data

    def _transform_ee_pose(self, data):
        if data is None:
            return
        for tf in data.transforms:
            for i in range(2, 7):
                if tf.child_frame_id == 'link{}'.format(i):
                    trans = tf.transform.translation
                    quat = tf.transform.rotation
                    tf_link = pt.transform_from_pq(
                        [trans.x, trans.y, trans.z, quat.x, quat.y, quat.z, quat.w])
                    self.tm.add_transform('link{}'.format(i),
                                    'link{}'.format(i-1), tf_link)
                elif tf.child_frame_id == 'link1':
                    trans = tf.transform.translation
                    quat = tf.transform.rotation
                    tf_link = pt.transform_from_pq(
                        [trans.x, trans.y, trans.z, quat.x, quat.y, quat.z, quat.w])
                    self.tm.add_transform('link1', 'base_link', tf_link)

        end_effector_matrix = self.tm.get_transform('link6', 'base_link')
        pos = end_effector_matrix[:3, 3]
        end_effector_rotation_matrix = end_effector_matrix[:3, :3]
        euler = R.from_dcm(end_effector_rotation_matrix).as_quat()
        ee_pose = np.concatenate([pos, euler])
        return ee_pose

if __name__ == "__main__":    
    get_pose = GetEEPose()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        ee_pose = get_pose.ee_pose()
        print("ee_pose: ", ee_pose)
        rate.sleep()
