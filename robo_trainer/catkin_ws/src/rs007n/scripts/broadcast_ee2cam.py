#!/usr/bin/env python
import os
import yaml
import rospy
import rospkg
import tf 
import tf2_ros
from geometry_msgs.msg import TransformStamped

if __name__ == '__main__':
    rospy.init_node('world2camera')
    camera_name = rospy.get_param('~CAMERA_NAME')
    calibration_file = os.path.join(rospkg.RosPack().get_path('rs007n_launch'),
        'config', 'ee2cam_' + camera_name[0:11] + '.yml')

    with open(calibration_file, 'r') as f:
        data = yaml.load(f)
        calib_trans = data['ee2cam']

    world2camera = tf2_ros.StaticTransformBroadcaster()
    static_tf = TransformStamped()
    static_tf.header.stamp = rospy.Time.now()
    static_tf.header.frame_id = 'link6'
    static_tf.child_frame_id = camera_name + '_color_optical_frame'
    static_tf.transform.translation.x = calib_trans[0]
    static_tf.transform.translation.y = calib_trans[1]
    static_tf.transform.translation.z = calib_trans[2]
    static_tf.transform.rotation.x = calib_trans[3]
    static_tf.transform.rotation.y = calib_trans[4]
    static_tf.transform.rotation.z = calib_trans[5]
    static_tf.transform.rotation.w = calib_trans[6]
    world2camera.sendTransform(static_tf)

    rospy.spin()