'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math

import torch


class HighwayFeedForward(torch.nn.Module):
    '''
    Highway Network
    '''

    def __init__(self, hidden_size, drop_rate):
        super(HighwayFeedForward, self).__init__()
        
        self.ff1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, hidden_size)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, input_):
        ''' 
        HT+X(1-T)
        '''
        hh = torch.relu(self.ff1(input_))
        tt = torch.sigmoid(self.ff2(input_))

        return self.drop(hh*tt+input_*(1-tt))
