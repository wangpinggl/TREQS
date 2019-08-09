'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class AttentionSelf(torch.nn.Module):

    def __init__(self, input_size, hidden_size,
                 device=torch.device("cpu")):
        '''
        implementation of self-attention.
        '''
        super().__init__()

        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_, mask=None):
        '''
        input vector: input_
        output:
            attn_: attention weights
            cv: context vector
        '''
        attn_ = torch.tanh(self.ff1(input_))
        attn_ = self.ff2(attn_).squeeze(2)
        if mask is not None:
            attn_ = attn_.masked_fill(mask == 0, -1e9)
        attn_ = torch.softmax(attn_, dim=1)
        ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)

        return attn_, ctx_vec
