'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch


def gelu(input_):
    '''
    GELU activation function.
    '''
    return input_ * 0.5 * (1.0 + torch.erf(input_ / math.sqrt(2.0)))
