#!/usr/bin/env python

import numpy as np
import math
import tf.transformations
import modern_robotics

class VisualServoing(object):
    def __init__(self):
        self._translation_only = False
        self._lambda = 0.5
        self._target_features_set = False

class PBVS(VisualServoing):
    def __init__(self):
        super(PBVS, self).__init__()
        self._target_feature_t = []
        self._target_feature_R = []

    def set_target_feature(self, t_input, R_input):
        '''
        Set PBVS target feature

        Input:  (object in desired camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''
        self._target_feature_t = np.array(t_input).flatten()
        self._target_feature_R = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        self._target_features_set = True

        print "\n"
        print "PBVS Set Target:"
        print "t:{}".format(self._target_feature_t)
        print "R:{}".format(self._target_feature_R)

    def _calculate_error(self, t_input, R_input):
        '''
        Caculate error based on the input pose and the target pose

        Input:  (object in current camera frame)
                t_input, 1x3 vector
                R_input, 1x4 vector, quaternion

        Output: Error, [t_err, R_err], 6x1
        '''
        
        t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]

        # see paragraph above Eq.13
        # of Chaumette, Francois, and Seth Hutchinson. "Visual servo control. I. Basic approaches."
        t_del = t_curr - self._target_feature_t
        R_del = np.dot(self._target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, _)= tf.transformations.rotation_from_matrix(R_del_homo)

        if self._translation_only:
            error = np.hstack((t_del, np.zeros(3)))
        else:
            error = np.hstack((t_del, theta*u))
        
        # print "Error:{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(error[0], error[1], error[2], error[3], error[4], error[5])

        return error

    def _L(self, t_input, R_input):
        '''
        form interaction matrix / feature jacobian base on current camera pose

        Input:  (object in current camera frame)
                t_input, 1x3 vector
                R_input, 1x4 vector, quaternion

        Output: Interation Matrix (feature Jacobian), 6x6
        '''

        t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        
        R_del = np.dot(self._target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))

        (theta, u, _)= tf.transformations.rotation_from_matrix(R_del_homo)
        
        skew_symmetric = modern_robotics.VecToso3

        skew_t = skew_symmetric(t_curr)
        L_theta_u = np.identity(3) - (theta/2)*np.array(skew_symmetric(u)) + (1-(np.sinc(theta)/((np.sinc(theta/2))**2)))*np.dot(np.array(skew_symmetric(u)), np.array(skew_symmetric(u)))

        L_top = np.hstack((-np.identity(3), skew_t))
        L_bottom = np.hstack((np.zeros((3,3)), L_theta_u))

        L_out = np.vstack((L_top, L_bottom))

        return L_out

    def caculate_vel(self, t_input, R_input):
        '''
        calculate the twist of camera frame required to reach target pose

        Input:  (object in current camera frame)
                t_input, 1x3 vector
                R_input, 1x4 vector, quaternion

        Output: Twist in camera frame
                [nu_c, omg_c], 1x6
        '''

        L = self._L(t_input, R_input)
        error = self._calculate_error(t_input, R_input)

        vel = -self._lambda * np.dot(np.linalg.pinv(L), error)

        # t_curr = np.array(t_input).flatten()
        # R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        
        # R_del = np.dot(self._target_feature_R, R_curr.T)
        # R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))

        # (theta, u, _)= tf.transformations.rotation_from_matrix(R_del_homo)

        # skew_symmetric = modern_robotics.VecToso3

        # vel_trans = -self._lambda * ((self._target_feature_t - t_curr) + np.dot(skew_symmetric(t_curr), theta*u))
        # vel_rot = -self._lambda * theta * u

        # vel = np.concatenate((vel_rot, vel_trans))

        # print "Camera Twist:{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(vel[0], vel[1], vel[2], vel[3], vel[4], vel[5])

        return vel

    def _calculate_error2(self, t_input, R_input):
        '''
        Caculate error based on the input pose and the target pose

        Input:  (object in current camera frame)
                t_input, 1x3 vector
                R_input, 1x4 vector, quaternion
        '''
        
        t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]

        # see paragraph above Eq.13
        # of Chaumette, Francois, and Seth Hutchinson. "Visual servo control. I. Basic approaches."
        # t_del = t_curr - self._target_feature_t
        R_del = np.dot(self._target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, _)= tf.transformations.rotation_from_matrix(R_del_homo)


        # see paragraph above Eq.17
        # of Chaumette, Francois, and Seth Hutchinson. "Visual servo control. I. Basic approaches."
        t_del = self._target_feature_R*(-1 * np.dot(R_curr.T, t_curr)) + self._target_feature_t

        if self._translation_only:
            error = np.hstack((t_del, np.zeros((1,3))))
        else:
            error = np.hstack((t_del, theta*u))
        
        return error

    def _L2(self, t_input, R_input):
        '''
        form interaction matrix / feature jacobian base on current camera pose

        Input:  (object in current camera frame)
                t_input, 1x3 vector
                R_input, 1x4 vector, quaternion
        '''

        # t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        
        R_del = np.dot(self._target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, _)= tf.transformations.rotation_from_matrix(R_del_homo)

        skew_symmetric = modern_robotics.VecToso3

        # skew_t = skew_symmetric(t_curr)
        L_theta_u = np.identity(3) - (theta/2)*skew_symmetric(u) + (1-(np.sinc(theta)/(np.sinc(theta/2)**2)))*np.dot(skew_symmetric(u),skew_symmetric(u))

        L_top = np.hstack((R_del, np.zeros((3,3))))
        L_bottom = np.hstack((np.zeros((3,3)), L_theta_u))

        L_out = np.vstack((L_top, L_bottom))

        return L_out

    def caculate_vel2(self, t_input, R_input):
        '''
        calculate the velocity required to reach target pose

        Input:  (object in current camera frame)
                t_input, 1x3 vector
                R_input, 1x4 vector, quaternion
        '''

        #L = self._L2(t_input, R_input)

        #error = 0