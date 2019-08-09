'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math

import torch


class LayerNormalization(torch.nn.Module):
    '''
    Epsilon outsize the square root.
    '''

    def __init__(self, size, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(size))
        self.beta = torch.nn.Parameter(torch.zeros(size))
        self.eps = eps

        self.register_parameter('gamma', self.gamma)
        self.register_parameter('beta', self.beta)

    def forward(self, input_):
        mean = torch.mean(input_, -1, keepdim=True)
        std = torch.std(input_, -1, keepdim=True)

        return self.gamma*(input_-mean)/(std+self.eps)+self.beta
