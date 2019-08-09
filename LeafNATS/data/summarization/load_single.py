'''
@author Tian Shi
Please contact tshi@vt.edu

Users need to rewrite this file based on their data format.
'''
import glob
import os
import random
import re
import shutil

import numpy as np
import torch
from torch.autograd import Variable


def process_minibatch(batch_id, path_,
                      fkey_, batch_size,
                      vocab2id, max_lens=[400, 100]):
    '''
    Process the minibatch.
    summary<sec>article.
    '''
    file_ = os.path.join(path_, 'batch_{}_{}'.format(
        fkey_, batch_size), str(batch_id))
    fp = open(file_, 'r')
    src_arr = []
    trg_arr = []
    src_lens = []
    trg_lens = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dabs = re.split(r'\s', arr[0])
        dabs = list(filter(None, dabs)) + ['<stop>']
        trg_lens.append(len(dabs))

        dabs2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dabs
        ]
        trg_arr.append(dabs2id)

        dart = re.split(r'\s', arr[1])
        dart = list(filter(None, dart))
        src_lens.append(len(dart))
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart
        ]
        src_arr.append(dart2id)
    fp.close()

    src_max_lens = max_lens[0]
    trg_max_lens = max_lens[1]

    src_arr = [itm[:src_max_lens] for itm in src_arr]
    trg_arr = [itm[:trg_max_lens] for itm in trg_arr]

    src_arr = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr
    ]
    trg_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
        for itm in trg_arr
    ]
    trg_output_arr = [
        itm[1:] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
        for itm in trg_arr
    ]

    src_var = Variable(torch.LongTensor(src_arr))
    trg_input_var = Variable(torch.LongTensor(trg_input_arr))
    trg_output_var = Variable(torch.LongTensor(trg_output_arr))

    return src_var, trg_input_var, trg_output_var


def process_minibatch_explicit(batch_id, path_,
                               fkey_, batch_size,
                               vocab2id, max_lens=[400, 100]):
    '''
    Process the minibatch. 
    OOV explicit, i.e., build extended vocabulary.
    '''

    file_ = os.path.join(path_, 'batch_{}_{}'.format(
        fkey_, batch_size), str(batch_id))
    # build extended vocabulary
    fp = open(file_, 'r')
    ext_vocab = {}
    ext_id2oov = {}
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dabs = re.split(r'\s', arr[0])
        dabs = list(filter(None, dabs))
        for wd in dabs:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
        dart = re.split(r'\s', arr[1])
        dart = list(filter(None, dart))
        for wd in dart:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
    cnt = len(vocab2id)
    for wd in ext_vocab:
        ext_vocab[wd] = cnt
        ext_id2oov[cnt] = wd
        cnt += 1
    fp.close()

    fp = open(file_, 'r')
    src_arr = []
    src_arr_ex = []
    trg_arr = []
    trg_arr_ex = []
    src_lens = []
    trg_lens = []
    for line in fp:
        # abstract
        arr = re.split('<sec>', line[:-1])
        dabs = re.split(r'\s', arr[0])
        dabs = list(filter(None, dabs)) + ['<stop>']
        trg_lens.append(len(dabs))
        # UNK
        dabs2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dabs
        ]
        trg_arr.append(dabs2id)
        # extend vocab
        dabs2id = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dabs
        ]
        trg_arr_ex.append(dabs2id)
        # article
        dart = re.split(r'\s', arr[1])
        dart = list(filter(None, dart))
        src_lens.append(len(dart))
        # UNK
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart
        ]
        src_arr.append(dart2id)
        # extend vocab
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dart
        ]
        src_arr_ex.append(dart2id)
    fp.close()

    src_max_lens = max_lens[0]
    trg_max_lens = max_lens[1]

    src_arr = [itm[:src_max_lens] for itm in src_arr]
    trg_arr = [itm[:trg_max_lens] for itm in trg_arr]
    src_arr_ex = [itm[:src_max_lens] for itm in src_arr_ex]
    trg_arr_ex = [itm[:trg_max_lens] for itm in trg_arr_ex]

    src_arr = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr
    ]
    trg_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
        for itm in trg_arr
    ]
    # extend oov
    src_arr_ex = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr_ex
    ]
    trg_output_arr_ex = [
        itm[1:] + [vocab2id['<pad>']]*(1+trg_max_lens-len(itm))
        for itm in trg_arr_ex
    ]

    src_var = Variable(torch.LongTensor(src_arr))
    trg_input_var = Variable(torch.LongTensor(trg_input_arr))
    # extend oov
    src_var_ex = Variable(torch.LongTensor(src_arr_ex))
    trg_output_var_ex = Variable(torch.LongTensor(trg_output_arr_ex))

    return ext_id2oov, src_var, trg_input_var, \
        src_var_ex, trg_output_var_ex


def process_minibatch_test(batch_id, path_,
                           fkey_, batch_size,
                           vocab2id, src_lens):
    '''
    Process the minibatch test
    '''

    file_ = os.path.join(path_, 'batch_{}_{}'.format(
        fkey_, batch_size), str(batch_id))
    fp = open(file_, 'r')
    src_arr = []
    src_idx = []
    src_wt = []
    trg_arr = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dabs = re.split(r'\s', arr[0])
        dabs = list(filter(None, dabs))
        dabs = ' '.join(dabs)
        trg_arr.append(dabs)

        dart = re.split(r'\s', arr[1])
        dart = list(filter(None, dart))
        src_arr.append(dart)
        dart2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                   for wd in dart]
        src_idx.append(dart2id)
        dart2wt = [0.0 if wd in vocab2id else 1.0 for wd in dart]
        src_wt.append(dart2wt)
    fp.close()

    src_idx = [itm[:src_lens] for itm in src_idx]
    src_idx = [itm + [vocab2id['<pad>']] *
               (src_lens-len(itm)) for itm in src_idx]
    src_var = Variable(torch.LongTensor(src_idx))

    src_wt = [itm[:src_lens] for itm in src_wt]
    src_wt = [itm + [0.0]*(src_lens-len(itm)) for itm in src_wt]
    src_msk = Variable(torch.FloatTensor(src_wt))

    src_arr = [itm[:src_lens] for itm in src_arr]
    src_arr = [itm + ['<pad>']*(src_lens-len(itm)) for itm in src_arr]

    return src_var, src_arr, src_msk, trg_arr


def process_minibatch_explicit_test(batch_id, path_,
                                    fkey_, batch_size,
                                    vocab2id, src_lens):
    '''
    Process the minibatch test. 
    OOV explicit.
    '''

    file_ = os.path.join(path_, 'batch_{}_{}'.format(
        fkey_, batch_size), str(batch_id))
    # build extended vocabulary
    fp = open(file_, 'r')
    ext_vocab = {}
    ext_id2oov = {}
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dart = re.split(r'\s', arr[1])
        dart = list(filter(None, dart))
        for wd in dart:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
    cnt = len(vocab2id)
    for wd in ext_vocab:
        ext_vocab[wd] = cnt
        ext_id2oov[cnt] = wd
        cnt += 1
    fp.close()

    fp = open(file_, 'r')
    src_arr = []
    src_idx = []
    src_idx_ex = []
    src_wt = []
    trg_arr = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        dabs = re.split(r'\s', arr[0])
        dabs = list(filter(None, dabs))
        dabs = ' '.join(dabs)
        trg_arr.append(dabs)

        dart = re.split(r'\s', arr[1])
        dart = list(filter(None, dart))
        src_arr.append(dart)
        dart2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                   for wd in dart]
        src_idx.append(dart2id)
        dart2id = [vocab2id[wd] if wd in vocab2id else ext_vocab[wd]
                   for wd in dart]
        src_idx_ex.append(dart2id)
        dart2wt = [0.0 if wd in vocab2id else 1.0 for wd in dart]
        src_wt.append(dart2wt)
    fp.close()

    src_idx = [itm[:src_lens] for itm in src_idx]
    src_idx = [itm + [vocab2id['<pad>']] *
               (src_lens-len(itm)) for itm in src_idx]
    src_var = Variable(torch.LongTensor(src_idx))

    src_idx_ex = [itm[:src_lens] for itm in src_idx_ex]
    src_idx_ex = [itm + [vocab2id['<pad>']] *
                  (src_lens-len(itm)) for itm in src_idx_ex]
    src_var_ex = Variable(torch.LongTensor(src_idx_ex))

    src_wt = [itm[:src_lens] for itm in src_wt]
    src_wt = [itm + [0.0]*(src_lens-len(itm)) for itm in src_wt]
    src_msk = Variable(torch.FloatTensor(src_wt))

    src_arr = [itm[:src_lens] for itm in src_arr]
    src_arr = [itm + ['<pad>']*(src_lens-len(itm)) for itm in src_arr]

    return ext_id2oov, src_var, src_var_ex, src_arr, src_msk, trg_arr
