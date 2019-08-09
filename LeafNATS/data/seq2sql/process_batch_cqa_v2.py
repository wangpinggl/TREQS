'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import os
import random
import re

import numpy as np
import torch
from torch.autograd import Variable


def process_minibatch(
    input_,  # batch data
    vocab2id,  # {word: id}
    format_statistics,
    max_lens=50
):
    '''
    Process minibatch.
    {'question_refine_tok': content, 'logic_form': conent}
    '''
    logicForm = {}

    ques_arr = []
    sel_arr = []
    agg_arr = []
    cond_arr = []
    for line in input_:
        data = json.loads(line)
        # process questions
        ques = data['question_refine_tok']
        ques_arr.append([
            vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
            for wd in ques])
        # process logic form.
        sel_arr.append(data['format']['sel'])
        agg_arr.append(data['format']['agg_col'])
        cond_arr.append(data['format']['cond'])

    ques_arr = [itm[:max_lens] for itm in ques_arr]
    ques_arr = [itm + [vocab2id['<pad>']] *
                (max_lens-len(itm)) for itm in ques_arr]
    ques_var = Variable(torch.LongTensor(ques_arr))

    pad_mask = Variable(torch.FloatTensor(ques_arr))
    pad_mask[pad_mask != float(vocab2id['<pad>'])] = -1.0
    pad_mask[pad_mask == float(vocab2id['<pad>'])] = 0.0
    pad_mask = -1.0*pad_mask
    # aggregation operation
    sel_var = Variable(torch.LongTensor(sel_arr))
    logicForm['sel'] = sel_var
    # aggregation column
    total_cols = 0
    for ky in format_statistics['nColumns']:
        total_cols += format_statistics['nColumns'][ky]
    agg_var = Variable(torch.zeros(
        [len(sel_arr), total_cols], dtype=torch.long))
    for k, itm in enumerate(agg_arr):
        for sen in itm:
            try:
                last_idx = format_statistics['nColumns'][str(sen[0]-1)]
            except:
                last_idx = 0
            idx = last_idx + sen[1]
            agg_var[k, idx] = 1
    logicForm['agg_col'] = agg_var
    # condition columns
    cond_col_var = Variable(torch.zeros(
        [len(sel_arr), total_cols], dtype=torch.long))
    cond_op_var = Variable(torch.ones(
        [len(sel_arr), total_cols], dtype=torch.long))
    cond_op_var = -1*cond_op_var
    cond_val_var = Variable(torch.ones(
        [len(sel_arr), total_cols, max_lens], dtype=torch.long))
    cond_val_var = -1*cond_val_var
    for k, itm in enumerate(cond_arr):
        for sen in itm:
            try:
                last_idx = format_statistics['nColumns'][str(sen[0]-1)]
            except:
                last_idx = 0
            idx = last_idx + sen[1]
            cond_col_var[k, idx] = 1
            cond_op_var[k, idx] = sen[2]
            cond_val_arr = sen[-1][:max_lens]
            cond_val_arr = cond_val_arr + (max_lens-len(cond_val_arr))*[-1]
            cond_val_arr = Variable(torch.LongTensor(cond_val_arr))
            cond_val_var[k, idx] = cond_val_arr
    logicForm['cond_col'] = cond_col_var
    logicForm['cond_op'] = cond_op_var
    logicForm['cond_val'] = cond_val_var

    return ques_var, pad_mask, logicForm
