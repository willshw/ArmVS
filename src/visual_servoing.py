#!/usr/bin/env python

import numpy as np
import math
import tf.transformations

from util import *

class VisualServoing(object):
    def __init__(self):
        self.__translation_only = False
        self.__lambda = 0.5
        self.__target_features_set = False

class PBVS(VisualServoing):
    def __init__(self):
        super(PBVS, self).__init__()
        self.__target_feature_t = []
        self.__target_feature_R = []

    def set_target_feature(self, t_input, R_input):
        '''
        Set PBVS target feature

        Input:  (object in desired camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''
        self.__target_feature_t = np.array(t_input).flatten()
        self.__target_feature_R = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        self.__target_features_set = True

        print "PBVS Set Target, t:{0}, R:{1}".format(self.__target_feature_t, self.__target_feature_R)

    def __calculate_error(self, t_input, R_input):
        '''
        Caculate error based on the input pose and the target pose

        Input:  (object in current camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''
        
        t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]

        # see paragraph above Eq.13
        # of Chaumette, François, and Seth Hutchinson. "Visual servo control. I. Basic approaches."
        t_del = t_curr - self.__target_feature_t
        R_del = np.dot(self.__target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, p)= tf.transformations.rotation_from_matrix(R_del_homo)

        if self.__translation_only:
            error = np.hstack((t_del, np.zeros((1,3))))
        else:
            error = np.hstack((t_del, theta*u))
        
        return error

    def __L(self, t_input, R_input):
        '''
        form interaction matrix / feature jacobian base on current camera pose

        Input:  (object in current camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''

        t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        
        R_del = np.dot(self.__target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, p)= tf.transformations.rotation_from_matrix(R_del_homo)

        skew_t = skew_symmetric(t_curr)
        L_theta_u = np.identity(3) - (theta/2)*skew_symmetric(u) + (1-(np.sinc(theta)/(np.sinc(theta/2)**2)))*np.dot(skew_symmetric(u),skew_symmetric(u))

        L_top = np.hstack((-np.identity(3), skew_t))
        L_bottom = np.hstack((np.zeros((3,3)), L_theta_u))

        L_out = np.vstack((L_top, L_bottom))

        return L_out

    def caculate_vel(self, t_input, R_input):
        '''
        calculate the velocity required to reach target pose

        Input:  (object in current camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''

        L = self.__L(t_input, R_input)
        error = self.__calculate_error(t_input, R_input)

        vel = -self.__lambda * np.dot(np.linalg.pinv(L), error)
        
        return vel

    def __calculate_error2(self, t_input, R_input):
        '''
        Caculate error based on the input pose and the target pose

        Input:  (object in current camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''
        
        t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]

        # see paragraph above Eq.13
        # of Chaumette, François, and Seth Hutchinson. "Visual servo control. I. Basic approaches."
        # t_del = t_curr - self.__target_feature_t
        R_del = np.dot(self.__target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, p)= tf.transformations.rotation_from_matrix(R_del_homo)


        # see paragraph above Eq.17
        # of Chaumette, François, and Seth Hutchinson. "Visual servo control. I. Basic approaches."
        t_del = self.__target_feature_R*(-1 * np.dot(R_curr.T, t_curr)) + self.__target_feature_t

        if self.__translation_only:
            error = np.hstack((t_del, np.zeros((1,3))))
        else:
            error = np.hstack((t_del, theta*u))
        
        return error

    def __L2(self, t_input, R_input):
        '''
        form interaction matrix / feature jacobian base on current camera pose

        Input:  (object in current camera frame)
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''

        # t_curr = np.array(t_input).flatten()
        R_curr = tf.transformations.quaternion_matrix(R_input)[0:3, 0:3]
        
        R_del = np.dot(self.__target_feature_R, R_curr.T)
        R_del_homo = np.vstack((np.hstack((R_del, np.zeros((3,1)))), np.array([0, 0, 0, 1])))
        (theta, u, p)= tf.transformations.rotation_from_matrix(R_del_homo)

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
                t_input, 3x1 vector
                R_input, 4x1 vector, quaternion
        '''

        L = self.__L2(t_input, R_input)



        error = 