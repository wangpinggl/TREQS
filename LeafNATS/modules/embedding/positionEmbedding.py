'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math

import torch


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, max_len,
                 hidden_size, drop_rate):
        super().__init__()

        # Compute the positional embeddings once in log space.
        posEmb = torch.zeros(max_len, hidden_size)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) *
                             -(math.log(10000.0) / hidden_size))

        posEmb[:, 0::2] = torch.sin(position * div_term)
        posEmb[:, 1::2] = torch.cos(position * div_term)
        posEmb = posEmb.unsqueeze(0)
        self.register_buffer('posEmb', posEmb)

        self.drop = nn.Dropout(drop_rate)

    def forward(self, input_):

        output_ = input_ + Variable(
            self.posEmb[:, :input_.size(1)], requires_grad=False)

        return self.dropout(output_)
