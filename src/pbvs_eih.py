#!/usr/bin/env python

import sys
import rospy
import roslib
import numpy as np

from tf.transformations import *
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
        baxter = BaxterVS(limb, tf_listener)
        controller = PBVS()

        # set target pose
        try:
            (R_target,t_target) = tf_listener.lookupTransform('/' + limb + '_hand_camera', '/goal', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print 'Error! Cannot find [goal] to [{}_hand_camera] tf.'.format(limb)
            sys.exit(0)

        controller.set_target_feature(t_target, R_target)

        # set up baxter
        while not rospy.is_shutdown():

            # get pose estimation from apriltag node
            try:
                (R_curr,t_curr) = tf_listener.lookupTransform('/' + limb + '_hand_camera', '/tag_0', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print 'Warning! Cannot find [tag_0] to [{}_hand_camera] tf.'.format(limb)
                continue

            # perform visual servoing
            vel_cam = controller.caculate_vel(t_curr, R_curr)

            vel_body = BaxterVS.body_frame_twist(vel_cam)
            BaxterVS.set_joint_vel(vel_body)

            r.sleep()

    except rospy.ROSInterruptException:
        pass