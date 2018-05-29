#!/usr/bin/env python

import numpy

def skew_symmetric(v):
    '''
    return skew symmetric matrix of vector v
    '''
    return [[0,      -v[2],  v[1]],
            [v[2],       0, -v[0]],
            [-v[1], v[0],       0]]
