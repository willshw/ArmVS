#!/usr/bin/env python

import sys
import rospy
import numpy as np

# from tf.transformations import *
import tf

from baxter import BaxterVS
from visual_servoing import PBVS

from std_msgs.msg import (
    Header,
    UInt16,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

if __name__ == '__main__':

    try:
        rospy.init_node('pbvs_eih')
        
        r = rospy.Rate(60)
        tf_listener = tf.TransformListener()

        limb = 'left'
        baxter_vs = BaxterVS(limb, tf_listener)
        controller = PBVS()
        controller._translation_only = False

        # set target pose
        try:
            tf_listener.waitForTransform('/desired_camera_frame', '/tag_0', rospy.Time(), rospy.Duration(4.0))
            (t_target, R_target) = tf_listener.lookupTransform('/desired_camera_frame', '/tag_0', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print 'Error! Cannot find [tag_0] to [desired_camera_frame] tf.'
            sys.exit(0)

        controller.set_target_feature(t_target, R_target)

        # set up baxter
        while not rospy.is_shutdown():
            print "\n\nNew Loop"

            # get pose estimation from apriltag node
            try:
                # tf_listener.waitForTransform('/' + limb + '_hand_camera', '/tag_0', rospy.Time(), rospy.Duration(4.0))
                (t_curr, R_curr) = tf_listener.lookupTransform('/' + limb + '_hand_camera', '/tag_0', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print 'Warning! Cannot find [tag_0] to [{}_hand_camera] tf.'.format(limb)
                continue

            # perform visual servoing
            vel_cam = controller.caculate_vel(t_curr, R_curr)

            vel_cam = np.concatenate((vel_cam[3:6], vel_cam[0:3]))

            vel_body = baxter_vs.body_frame_twist(vel_cam)

            vel_body = np.concatenate((vel_body[3:6], vel_body[0:3]))

            baxter_vs.set_joint_vel(vel_body)

            r.sleep()

    except rospy.ROSInterruptException:
        pass