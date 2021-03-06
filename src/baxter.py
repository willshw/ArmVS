#!/usr/bin/env python

import baxter_interface
from baxter_pykdl import baxter_kinematics

import numpy as np
import roslib
import rospy
import sys
from tf.transformations import quaternion_matrix
import tf
import copy

import modern_robotics

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)


class BaxterVS(object):
    '''
    A wrapper class for visual servoing related functions for baxter arm
    '''
    
    def __init__(self, limb, tf_listener):
        
        self._limb = limb
        self._tf_listener = tf_listener

        # h: hand, c: camera
        try:
            self._tf_listener.waitForTransform('/' + self._limb + '_hand', '/' + self._limb + '_hand_camera', rospy.Time(), rospy.Duration(4.0))
            (self._t_hc, self._R_hc) = self._tf_listener.lookupTransform('/' + self._limb + '_hand', '/' + self._limb + '_hand_camera', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print 'Error! Cannot find [{}_hand_camera] to [{}_hand] tf.'.format(self._limb, self._limb)
            sys.exit(0)

        self._R_hc = quaternion_matrix(self._R_hc)
        
        # frame transforms from camera to hand, only look up once
        self._T_hc = modern_robotics.RpToTrans(self._R_hc[:3,:3], self._t_hc)
        self._Ad_hc = modern_robotics.Adjoint(self._T_hc)

        # frame transforms from hand to body, look up in every loop
        self._T_bh = np.zeros((4,4))
        self._Ad_bh = np.zeros((6,6))

        self._arm = baxter_interface.limb.Limb(self._limb)  
        self._kin = baxter_kinematics(self._limb)

    def update_hand_to_body_transforms(self):
        '''
        Update frame transforms for transformation matrix and twist
        '''

        try:
            self._tf_listener.waitForTransform('/base', '/' + self._limb + '_hand', rospy.Time(), rospy.Duration(4.0))
            (t, R) = self._tf_listener.lookupTransform('/base', '/' + self._limb + '_hand', rospy.Time(0))
        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print 'Warning! Cannot find [{}_hand] tf to [base].'.format(self._limb)

        R = quaternion_matrix(R)

        self._T_bh = modern_robotics.RpToTrans(R[:3,:3], t)
        self._Ad_bh = modern_robotics.Adjoint(self._T_bh)

    def body_frame_twist(self, v_c):
        '''
        Take a 6x1 twist vector in camera frame,
        returns the Adjoint from camera frame to body frame,
        for calculating a camera frame twist in body frame

        Input: twist, [nu, omg], 1x6

        Output: twist, [nu, omg], 1x6
        '''

        self.update_hand_to_body_transforms()

        v_c = np.concatenate((v_c[3:6], v_c[0:3]))
        
        v_b = np.dot(self._Ad_bh, np.dot(self._Ad_hc, v_c))

        v_b = np.concatenate((v_b[3:6], v_b[0:3]))

        return v_b
        
    def set_joint_vel(self, vel_b):
        '''
        Takes a 6x1 twist vector in body frame,
        sets the corresponding joint velocities using the PyKDL package.

        Input: twist, [nu, omg], 1x6
        '''

         # Calculate joint velocities to achieve desired velocity
        joint_vels = np.dot(self._kin.jacobian_pseudo_inverse(), vel_b)
        joint_vels = np.array(joint_vels).reshape(-1,)
    
        joints=dict(zip(self._arm.joint_names(), joint_vels))

        # print "Joint Vel Command:{}".format(joint_vels)

        self._arm.set_joint_velocities(joints)
