'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import json
import os
import random
import re
import shutil

import numpy as np
import torch
from torch.autograd import Variable


def process_batch(
        batch_id,  # current batch_id file name
        path_,  # path to data dir
        fkey_,  # train/val/test
        batch_size,
        vocab2id,
        max_lens=[50, 30]):
    '''
    Process the minibatch.
    {
        'question_refine_tok': [a list of tokens],
        'sql_tok': [a list of tokens]
    }
    '''
    file_ = os.path.join(path_, 'batch_{}_{}'.format(
        fkey_, batch_size), str(batch_id))
    output = {}
    # build extended vocabulary
    fp = open(file_, 'r')
    ext_vocab = {}
    ext_id2oov = {}
    for line in fp:
        arr = json.loads(line)
        if fkey_ == 'train' or fkey_ == 'validate':
            dabs = arr['sql_tok']
            for wd in dabs:
                if wd not in vocab2id:
                    ext_vocab[wd] = {}
        dart = arr['question_refine_tok']
        for wd in dart:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
    cnt = len(vocab2id)
    for wd in ext_vocab:
        ext_vocab[wd] = cnt
        ext_id2oov[cnt] = wd
        cnt += 1
    fp.close()
    output['ext_vocab'] = ext_vocab
    output['ext_id2oov'] = ext_id2oov
    # process data
    fp = open(file_, 'r')
    src_lens = []
    trg_lens = []
    for line in fp:
        # abstract
        arr = json.loads(line)
        dabs = arr['sql_tok']
        if fkey_ == 'train' or fkey_ == 'validate':
            dabs = list(filter(None, dabs)) + ['<stop>']
            try:
                output['trg_txt'].append(dabs)
            except:
                output['trg_txt'] = [dabs]
            trg_lens.append(len(dabs))
            # UNK
            dabs2id = [
                vocab2id[wd] if wd in vocab2id
                else vocab2id['<unk>']
                for wd in dabs]
            try:
                output['trg_idx'].append(dabs2id)
            except:
                output['trg_idx'] = [dabs2id]
            # extend vocab
            dabs2id_ex = [
                vocab2id[wd] if wd in vocab2id
                else ext_vocab[wd]
                for wd in dabs]
            try:
                output['trg_idx_ex'].append(dabs2id_ex)
            except:
                output['trg_idx_ex'] = [dabs2id_ex]
        else:
            try:
                output['trg_txt'].append(dabs)
            except:
                output['trg_txt'] = [dabs]
        # article
        dart = arr['question_refine_tok']
        src_lens.append(len(dart))
        try:
            output['src_txt'].append(dart)
        except:
            output['src_txt'] = [dart]
        # UNK
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart]
        try:
            output['src_idx'].append(dart2id)
        except:
            output['src_idx'] = [dart2id]
        # extend vocab
        dart2id_ex = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dart]
        try:
            output['src_idx_ex'].append(dart2id_ex)
        except:
            output['src_idx_ex'] = [dart2id_ex]
        # UNK mask
        dart2wt = [0.0 if wd in vocab2id else 1.0 for wd in dart]
        try:
            output['src_mask_unk'].append(dart2wt)
        except:
            output['src_mask_unk'] = [dart2wt]
    fp.close()

    src_max_lens = max_lens[0]
    trg_max_lens = max_lens[1]

    output['src_idx'] = [itm[:src_max_lens] for itm in output['src_idx']]
    output['src_idx'] = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in output['src_idx']]
    output['src_var'] = Variable(torch.LongTensor(output['src_idx']))

    output['src_idx_ex'] = [itm[:src_max_lens] for itm in output['src_idx_ex']]
    output['src_idx_ex'] = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in output['src_idx_ex']]
    output['src_var_ex'] = Variable(torch.LongTensor(output['src_idx_ex']))

    output['src_mask_unk'] = [
        itm[:src_max_lens] for itm in output['src_mask_unk']]
    output['src_mask_unk'] = [
        itm + [0.0]*(src_max_lens-len(itm))
        for itm in output['src_mask_unk']]
    output['src_mask_unk'] = Variable(
        torch.FloatTensor(output['src_mask_unk']))

    output['src_mask_pad'] = Variable(torch.FloatTensor(output['src_idx']))
    output['src_mask_pad'][output['src_mask_pad']
                           != float(vocab2id['<pad>'])] = -1.0
    output['src_mask_pad'][output['src_mask_pad']
                           == float(vocab2id['<pad>'])] = 0.0
    output['src_mask_pad'] = -1.0*output['src_mask_pad']

    if fkey_ == 'train' or fkey_ == 'validate':
        output['trg_idx'] = [itm[:trg_max_lens] for itm in output['trg_idx']]
        output['trg_input_idx'] = [
            itm[:-1] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
            for itm in output['trg_idx']]
        output['trg_input_var'] = Variable(
            torch.LongTensor(output['trg_input_idx']))

        output['trg_output_idx'] = [
            itm[1:] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
            for itm in output['trg_idx']]
        output['trg_output_var'] = Variable(
            torch.LongTensor(output['trg_output_idx']))

        output['trg_idx_ex'] = [itm[:trg_max_lens]
                                for itm in output['trg_idx_ex']]
        output['trg_output_idx_ex'] = [
            itm[1:] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
            for itm in output['trg_idx_ex']]
        output['trg_output_var_ex'] = Variable(
            torch.LongTensor(output['trg_output_idx_ex']))

    return output
