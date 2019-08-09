'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math

import torch

from LeafNATS.modules.attention.attention_multi_head import \
    MultiHeadedAttention
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization
from LeafNATS.modules.utils.PositionwiseFeedForward import \
    PositionwiseFeedForward


class TransformerBlock(torch.nn.Module):
    '''
    Implementation of Transformer
    '''

    def __init__(self, input_size,
                 n_heads, drop_rate,
                 device=torch.device("cpu")):
        super().__init__()
        # multi-head attention
        self.attentionMH = MultiHeadedAttention(n_heads, input_size, drop_rate)
        # layer normalization
        self.norm1 = LayerNormalization(input_size)
        self.norm2 = LayerNormalization(input_size)
        # layer feed-forward
        self.layer_ff = PositionwiseFeedForward(
            input_size, input_size*4, input_size, drop_rate)

        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, input_, mask=None):
        '''
        Transformer
        '''
        hd = self.attentionMH(input_, mask)
        hd = self.norm1(input_ + self.drop(hd))
        hd = self.norm2(hd + self.layer_ff(hd))

        return self.drop(hd)
