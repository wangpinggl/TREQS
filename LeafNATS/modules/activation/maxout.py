'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch


def maxout(input_, pool_size):
    '''
    maxout activation
    '''
    input_size = list(input_.size())
    assert input_.size(-1) % pool_size == 0

    out_size = input_.size(-1) // pool_size
    input_size[-1] = out_size
    input_size.append(pool_size)

    return input_.view(*input_size).max(-1)
