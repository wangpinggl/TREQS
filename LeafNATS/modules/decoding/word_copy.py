'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import sys
import time
import numpy as np

import torch
from torch.autograd import Variable


def word_copy(args,  beam_seq,
              beam_attn_,  src_msk,
              src_arr,  batch_size,
              id2vocab,  ext_id2oov):
    '''
    This is a meta-algorithm that can combine with any seq2seq models to replace OOV words.
    Copy word from source to summary based on attention weights.
    '''
    out_arr = []
    if args.copy_words:
        src_msk = src_msk.repeat(1, args.beam_size).view(
            src_msk.size(0), args.beam_size, args.src_seq_lens).unsqueeze(0)
        beam_attn_ = beam_attn_*src_msk
        beam_copy = beam_attn_.topk(1, dim=3)[1].squeeze(-1)
        beam_copy = beam_copy[:, :, 0].transpose(0, 1)
        wdidx_copy = beam_copy.data.cpu().numpy()
        for b in range(batch_size):
            gen_text = beam_seq.data.cpu().numpy()[b, 0]
            gen_text = [id2vocab[wd] if wd in id2vocab else ext_id2oov[wd]
                        for wd in gen_text]
            gen_text = gen_text[1:]
            for j in range(len(gen_text)):
                if gen_text[j] == '<unk>':
                    gen_text[j] = src_arr[b][wdidx_copy[b, j]]
            out_arr.append(' '.join(gen_text))
    else:
        for b in range(batch_size):
            gen_text = beam_seq.data.cpu().numpy()[b, 0]
            gen_text = [id2vocab[wd] if wd in id2vocab else ext_id2oov[wd]
                        for wd in gen_text]
            gen_text = gen_text[1:]
            out_arr.append(' '.join(gen_text))

    return out_arr
