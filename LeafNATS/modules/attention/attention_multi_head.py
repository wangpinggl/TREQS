'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math

import torch
import torch.nn.functional as F


class MultiHeadedAttention(torch.nn.Module):
    '''
    Implement of multi-head attention.
    '''

    def __init__(self, n_heads,
                 hidden_size, drop_rate):
        super().__init__()

        assert hidden_size % n_heads == 0
        self.n_dk = hidden_size // n_heads
        self.n_heads = n_heads

        self.proj_query = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_key = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_value = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(drop_rate)

        self.proj_output = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_, mask=None):
        '''
        Input: embedding.
        '''
        batch_size = input_.size(0)

        query = self.proj_query(input_)
        query = query.view(batch_size, -1, self.n_heads,
                           self.n_dk).transpose(1, 2)
        key = self.proj_key(input_)
        key = key.view(batch_size, -1, self.n_heads, self.n_dk).transpose(1, 2)
        value = self.proj_value(input_)
        value = value.view(batch_size, -1, self.n_heads,
                           self.n_dk).transpose(1, 2)

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.n_dk)
        if mask is not None:
            mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        cv = attn @ value
        cv = cv.transpose(1, 2)
        cv = cv.contiguous().view(batch_size, -1, self.n_heads*self.n_dk)

        return self.proj_output(cv)
