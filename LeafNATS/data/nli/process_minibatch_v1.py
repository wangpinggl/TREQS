'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import re

import torch
from torch.autograd import Variable


def process_minibatch(input_, vocab2id, 
                      premise_max_lens, 
                      hypothesis_max_lens):
    '''
    Process minibatch.
    '''
    len_premise = []
    len_hypothe = []
    premise_arr = []
    hypothe_arr = []
    label_arr = []
    for line in input_:
        data = json.loads(line)

        label_arr.append(data['gold_label']+1)

        premise = data['premise']
        len_premise.append(len(premise))
        premise2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                      for wd in premise]
        premise_arr.append(premise2id)

        hypothe = data['hypothesis']
        len_hypothe.append(len(hypothe))
        hypothe2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                      for wd in hypothe]
        hypothe_arr.append(hypothe2id)

    premise_lens = min(premise_max_lens, max(len_premise))
    hypothe_lens = min(hypothesis_max_lens, max(len_hypothe))

    premise_arr = [itm[:premise_lens] for itm in premise_arr]
    premise_arr = [itm + [vocab2id['<pad>']] *
                   (premise_lens-len(itm)) for itm in premise_arr]
    premise_var = Variable(torch.LongTensor(premise_arr))

    hypothe_arr = [itm[:hypothe_lens] for itm in hypothe_arr]
    hypothe_arr = [itm + [vocab2id['<pad>']] *
                   (hypothe_lens-len(itm)) for itm in hypothe_arr]
    hypothe_var = Variable(torch.LongTensor(hypothe_arr))

    label_var = Variable(torch.LongTensor(label_arr))

    premise_mask = Variable(torch.FloatTensor(premise_arr))
    premise_mask[premise_mask != 1.0] = 0.0
    premise_mask = 1.0 - premise_mask

    hypothe_mask = Variable(torch.FloatTensor(hypothe_arr))
    hypothe_mask[hypothe_mask != 1.0] = 0.0
    hypothe_mask = 1.0 - hypothe_mask

    return premise_var, hypothe_var, premise_mask, hypothe_mask, label_var
