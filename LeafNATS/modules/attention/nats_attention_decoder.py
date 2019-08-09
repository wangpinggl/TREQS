'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class AttentionDecoder(torch.nn.Module):
    '''
    Intra-decoder

    Paulus, R., Xiong, C., & Socher, R. (2017). 
    A deep reinforced model for abstractive summarization. 
    arXiv preprint arXiv:1705.04304.
    '''

    def __init__(self, hidden_size,  attn_method):
        '''
        hidden_size, # decoder hidden dimension
        attn_method # alignment method
        '''
        super().__init__()

        self.method = attn_method.lower()
        self.hidden_size = hidden_size

        if self.method == 'luong_concat':
            self.attn_en_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True)
            self.attn_de_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False)
            self.attn_warp_in = torch.nn.Linear(
                self.hidden_size, 1, bias=False)
        if self.method == 'luong_general':
            self.attn_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False)

    def forward(self,  dehy,  past_hy):
        '''
        dehy - current decoder hidden states
        past_hy - previous decoder hidden states
        '''
        # attention score
        if self.method == 'luong_concat':
            attn_agg = self.attn_en_in(
                past_hy) + self.attn_de_in(dehy.unsqueeze(1))
            attn_agg = torch.tanh(attn_agg)
            attn = self.attn_warp_in(attn_agg).squeeze(2)
        else:
            if self.method == 'luong_general':
                past_hy_new = self.attn_in(past_hy)
                attn = torch.bmm(past_hy_new, dehy.unsqueeze(2)).squeeze(2)
            else:
                attn = torch.bmm(past_hy, dehy.unsqueeze(2)).squeeze(2)
        attn = torch.softmax(attn, dim=1)
        # context vector
        attn2 = attn.unsqueeze(1)
        c_decoder = torch.bmm(attn2, past_hy).squeeze(1)

        return c_decoder, attn
