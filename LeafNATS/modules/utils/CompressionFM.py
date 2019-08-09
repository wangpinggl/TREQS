'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math

import torch


class CompressionFM(torch.nn.Module):
    '''
    FM layer
    '''

    def __init__(self,  input_size, fm_size):
        super(CompressionFM, self).__init__()
        self.LW = torch.nn.Linear(input_size, 1)
        self.QV = torch.nn.Parameter(torch.randn(input_size, fm_size))

    def forward(self, input_):
        ''' 
        Factor Machine Implementation.
        '''
        size_input = input_.size()
        input_ = input_.contiguous().view(-1, input_.size(-1))
        h0 = self.LW(input_)
        v1 = torch.mm(input_, self.QV)
        v1 = v1*v1
        v2 = torch.mm(input_*input_, self.QV*self.QV)
        vcat = torch.sum(v1 - v2, 1)

        fm = h0.squeeze() + 0.5*vcat
        fm = fm.view(size_input[0], size_input[1], 1)

        return fm
