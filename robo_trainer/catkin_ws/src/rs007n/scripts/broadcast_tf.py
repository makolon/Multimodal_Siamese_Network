#!/usr/bin/env python
import rospy

import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped

if __name__ == '__main__':
    rospy.init_node('tf_broadcast')

    camera_frame = tf2_ros.StaticTransformBroadcaster()
    static_tf_camera = TransformStamped()

    camera_name = rospy.get_param('~CAMERA_NAME')

    static_tf_camera.header.stamp = rospy.Time.now()

    # TODO: check frame_id
    static_tf_camera.header.frame_id = camera_name + '_color_optical_frame'
    static_tf_camera.child_frame_id = camera_name + '_link'

    static_tf_camera.transform.translation.x = 0.0147052564039
    static_tf_camera.transform.translation.y = 0.00012443851978
    static_tf_camera.transform.translation.z = 0.000286161684449
    static_tf_camera.transform.rotation.x = 0.506091553718
    static_tf_camera.transform.rotation.y = -0.496781819034
    static_tf_camera.transform.rotation.z = 0.499370179605
    static_tf_camera.transform.rotation.w = 0.4977032803

    camera_frame.sendTransform(static_tf_camera)
    rospy.spin()