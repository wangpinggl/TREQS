'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math

import torch


def gelu(x):
    '''
    GELU activation function.
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionwiseFeedForward(torch.nn.Module):
    '''
    FeedForward Neural Networks for each position
    '''

    def __init__(self, input_size,
                 hidden_size, output_size, drop_rate):
        super(PositionwiseFeedForward, self).__init__()
        
        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, output_size)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, input_):
        ''' 
        (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        '''
        return self.drop(self.ff2(gelu(self.ff1(input_))))
