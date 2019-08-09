'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderCNN(torch.nn.Module):

    def __init__(self, input_size,
                 kernel_size,  kernel_nums,
                 device=torch.device("cpu")):
        '''
        Implementation of CNN encoder.

            input_size,  # input_ dimension
            kernel_size,  # 3,4,5
            kernel_nums,  # 100, 200, 100
        '''
        super().__init__()

        kSize = re.split(',', kernel_size)
        kSize = [int(itm) for itm in kSize]
        kNums = re.split(',', kernel_nums)
        kNums = [int(itm) for itm in kNums]
        assert len(kSize) == len(kNums)

        self.convs1 = torch.nn.ModuleList([
            torch.nn.Conv2d(1, kNums[k], (kSize[k], input_size))
            for k in range(len(kNums))]).to(device)

    def forward(self, input_):
        '''
        input_: 
        '''
        input_ = input_.unsqueeze(1)
        h0 = [F.relu(conv(input_)).squeeze(3) for conv in self.convs1]
        h0 = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h0]
        h0 = torch.cat(h0, 1)

        return h0
