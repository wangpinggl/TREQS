'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class AttentionEncoder(torch.nn.Module):
    '''
    Bahdanau, D., Cho, K., & Bengio, Y. (2014). 
    Neural machine translation by jointly learning to align and translate. 
    arXiv preprint arXiv:1409.0473.
    Luong, M. T., Pham, H., & Manning, C. D. (2015). 
    Effective approaches to attention-based neural machine translation. 
    arXiv preprint arXiv:1508.04025.
    Paulus, R., Xiong, C., & Socher, R. (2017). 
    A deep reinforced model for abstractive summarization. 
    arXiv preprint arXiv:1705.04304.
    See, A., Liu, P. J., & Manning, C. D. (2017). 
    Get To The Point: Summarization with Pointer-Generator Networks. 
    arXiv preprint arXiv:1704.04368.
    '''

    def __init__(self, src_hidden_size, trg_hidden_size,
                 attn_method, repetition, 
                 src_hidden_doubled=True):
        '''
        src_hidden_size, # source side hidden dimension
        trg_hidden_size, # target side hidden dimension
        attn_method, # attention method
        repetition # approaches handle repetition
        '''
        super().__init__()
        self.method = attn_method.lower()
        self.repetition = repetition

        if self.method == 'luong_concat':
            if src_hidden_doubled:
                self.attn_en_in = torch.nn.Linear(
                    src_hidden_size*2, trg_hidden_size)
            else:
                self.attn_en_in = torch.nn.Linear(
                    src_hidden_size, trg_hidden_size)
            self.attn_de_in = torch.nn.Linear(
                trg_hidden_size, trg_hidden_size, bias=False)
            self.attn_cv_in = torch.nn.Linear(
                1, trg_hidden_size, bias=False)
            self.attn_warp_in = torch.nn.Linear(
                trg_hidden_size, 1, bias=False)
        if self.method == 'luong_general':
            if src_hidden_doubled:
                self.attn_in = torch.nn.Linear(
                    src_hidden_size*2, trg_hidden_size, bias=False)
            else:
                self.attn_in = torch.nn.Linear(
                    src_hidden_size, trg_hidden_size, bias=False)

    def forward(self, dehy, enhy, past_attn, src_mask=None):
        '''
        dehy,  # current decoder hidden state
        enhy,  # encoder hidden states
        past_attn,  # accumulate of previous attention.
        '''
        # attention score
        if self.method == 'luong_concat':
            attn_agg = self.attn_en_in(
                enhy) + self.attn_de_in(dehy.unsqueeze(1))
            if self.repetition[:4] == 'asee':
                attn_agg = attn_agg + self.attn_cv_in(past_attn.unsqueeze(2))
            attn_agg = torch.tanh(attn_agg)
            attn_ee = self.attn_warp_in(attn_agg).squeeze(2)
        else:
            if self.method == 'luong_general':
                enhy_new = self.attn_in(enhy)
                attn_ee = torch.bmm(enhy_new, dehy.unsqueeze(2)).squeeze(2)
            else:
                attn_ee = torch.bmm(enhy, dehy.unsqueeze(2)).squeeze(2)
        if src_mask is not None:
            attn_ee = attn_ee.masked_fill(torch.abs(src_mask) == 0, -1e20)
        # repetition and attention weights
        if self.repetition == 'temporal':
            attn_ee = torch.exp(attn_ee)
            if src_mask is not None:
                past_attn = past_attn.masked_fill(past_attn == 0, 1)
            attn = attn_ee/past_attn
            nm = torch.norm(attn, 1, 1).unsqueeze(1)
            if src_mask is not None:
                nm = nm.masked_fill(nm == 0, 1)
            attn = attn/nm
        else:
            attn = torch.softmax(attn_ee, dim=1)
        # context vector
        attn2 = attn.unsqueeze(1)
        c_encoder = torch.bmm(attn2, enhy).squeeze(1)

        return c_encoder, attn, attn_ee
