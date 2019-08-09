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
                      vocab2id, max_lens=[20, 100, 400]):
    '''
    Process the minibatch for newsroom.
    Multi-task.
    headline<sec>summary<sec>article.
    '''

    file_ = os.path.join(path_, 'batch_{}_{}'.format(
        fkey_, batch_size), str(batch_id))
    fp = open(file_, 'r')
    src_arr = []
    sum_arr = []
    ttl_arr = []
    src_lens = []
    sum_lens = []
    ttl_lens = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])

        dttl = re.split(r'\s', arr[0])
        dttl = list(filter(None, dttl)) + ['<stop>']
        ttl_lens.append(len(dttl))
        dttl2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dttl
        ]
        ttl_arr.append(dttl2id)

        dsum = re.split(r'\s', arr[1])
        dsum = list(filter(None, dsum)) + ['<stop>']
        sum_lens.append(len(dsum))
        dsum2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dsum
        ]
        sum_arr.append(dsum2id)

        dsrc = re.split(r'\s', arr[2])
        dsrc = list(filter(None, dsrc))
        src_lens.append(len(dsrc))
        dsrc2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dsrc
        ]
        src_arr.append(dsrc2id)
    fp.close()

    ttl_max_lens = max_lens[0]
    sum_max_lens = max_lens[1]
    src_max_lens = max_lens[2]

    ttl_arr = [itm[:ttl_max_lens] for itm in ttl_arr]
    sum_arr = [itm[:sum_max_lens] for itm in sum_arr]
    src_arr = [itm[:src_max_lens] for itm in src_arr]

    ttl_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+ttl_max_lens-len(itm))
        for itm in ttl_arr
    ]
    ttl_output_arr = [
        itm[1:] + [vocab2id['<pad>']]*(1+ttl_max_lens-len(itm))
        for itm in ttl_arr
    ]
    sum_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+sum_max_lens-len(itm))
        for itm in sum_arr
    ]
    sum_output_arr = [
        itm[1:] + [vocab2id['<pad>']]*(1+sum_max_lens-len(itm))
        for itm in sum_arr
    ]
    src_arr = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr
    ]

    ttl_input_var = Variable(torch.LongTensor(ttl_input_arr))
    ttl_output_var = Variable(torch.LongTensor(ttl_output_arr))
    sum_input_var = Variable(torch.LongTensor(sum_input_arr))
    sum_output_var = Variable(torch.LongTensor(sum_output_arr))
    src_var = Variable(torch.LongTensor(src_arr))

    return ttl_input_var, ttl_output_var, sum_input_var, sum_output_var, src_var


def process_minibatch_explicit(batch_id, path_,
                               fkey_, batch_size,
                               vocab2id, max_lens=[20, 100, 400]):
    '''
    Process the minibatch. 
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
        dttl = re.split(r'\s', arr[0])
        dttl = list(filter(None, dttl))
        for wd in dttl:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
        dsum = re.split(r'\s', arr[1])
        dsum = list(filter(None, dsum))
        for wd in dsum:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
        dsrc = re.split(r'\s', arr[2])
        dsrc = list(filter(None, dsrc))
        for wd in dsrc:
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
    sum_arr = []
    sum_arr_ex = []
    ttl_arr = []
    ttl_arr_ex = []
    src_lens = []
    sum_lens = []
    ttl_lens = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        # title
        dttl = re.split(r'\s', arr[0])
        dttl = list(filter(None, dttl)) + ['<stop>']
        ttl_lens.append(len(dttl))
        # UNK
        dttl2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dttl
        ]
        ttl_arr.append(dttl2id)
        # extend vocab
        dttl2id = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dttl
        ]
        ttl_arr_ex.append(dttl2id)
        # summary
        dsum = re.split(r'\s', arr[1])
        dsum = list(filter(None, dsum)) + ['<stop>']
        sum_lens.append(len(dsum))
        # UNK
        dsum2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dsum
        ]
        sum_arr.append(dsum2id)
        # extend vocab
        dsum2id = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dsum
        ]
        sum_arr_ex.append(dsum2id)
        # article
        dsrc = re.split(r'\s', arr[2])
        dsrc = list(filter(None, dsrc))
        src_lens.append(len(dsrc))
        # UNK
        dsrc2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dsrc
        ]
        src_arr.append(dsrc2id)
        # extend vocab
        dsrc2id = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dsrc
        ]
        src_arr_ex.append(dsrc2id)
    fp.close()

    ttl_max_lens = max_lens[0]
    sum_max_lens = max_lens[1]
    src_max_lens = max_lens[2]

    src_arr = [itm[:src_max_lens] for itm in src_arr]
    sum_arr = [itm[:sum_max_lens] for itm in sum_arr]
    ttl_arr = [itm[:ttl_max_lens] for itm in ttl_arr]
    src_arr_ex = [itm[:src_max_lens] for itm in src_arr_ex]
    sum_arr_ex = [itm[:sum_max_lens] for itm in sum_arr_ex]
    ttl_arr_ex = [itm[:ttl_max_lens] for itm in ttl_arr_ex]

    src_arr = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr
    ]
    sum_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+sum_max_lens-len(itm))
        for itm in sum_arr
    ]
    ttl_input_arr = [
        itm[:-1] + [vocab2id['<pad>']]*(1+ttl_max_lens-len(itm))
        for itm in ttl_arr
    ]
    # extend oov
    src_arr_ex = [
        itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
        for itm in src_arr_ex
    ]
    sum_output_arr_ex = [
        itm[1:] + [vocab2id['<pad>']]*(1+sum_max_lens-len(itm))
        for itm in sum_arr_ex
    ]
    ttl_output_arr_ex = [
        itm[1:] + [vocab2id['<pad>']]*(1+ttl_max_lens-len(itm))
        for itm in ttl_arr_ex
    ]

    src_var = Variable(torch.LongTensor(src_arr))
    sum_input_var = Variable(torch.LongTensor(sum_input_arr))
    ttl_input_var = Variable(torch.LongTensor(ttl_input_arr))
    # extend oov
    src_var_ex = Variable(torch.LongTensor(src_arr_ex))
    sum_output_var_ex = Variable(torch.LongTensor(sum_output_arr_ex))
    ttl_output_var_ex = Variable(torch.LongTensor(ttl_output_arr_ex))

    return ext_id2oov, ttl_input_var, sum_input_var, src_var, \
        ttl_output_var_ex, sum_output_var_ex, src_var_ex


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
    sum_arr = []
    ttl_arr = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])

        dttl = re.split(r'\s', arr[0])
        dttl = list(filter(None, dttl))
        dttl = ' '.join(dttl)
        ttl_arr.append(dttl)

        dsum = re.split(r'\s', arr[1])
        dsum = list(filter(None, dsum))
        dsum = ' '.join(dsum)
        sum_arr.append(dsum)

        dsrc = re.split(r'\s', arr[2])
        dsrc = list(filter(None, dsrc))
        src_arr.append(dsrc)
        dsrc2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                   for wd in dsrc]
        src_idx.append(dsrc2id)
        dsrc2wt = [0.0 if wd in vocab2id else 1.0 for wd in dsrc]
        src_wt.append(dsrc2wt)
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

    return src_var, src_arr, src_msk, sum_arr, ttl_arr


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
        dsrc = re.split(r'\s', arr[2])
        dsrc = list(filter(None, dsrc))
        for wd in dsrc:
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
    sum_arr = []
    ttl_arr = []
    for line in fp:
        arr = re.split('<sec>', line[:-1])

        dttl = re.split(r'\s', arr[0])
        dttl = list(filter(None, dttl))
        dttl = ' '.join(dttl)
        ttl_arr.append(dttl)

        dsum = re.split(r'\s', arr[1])
        dsum = list(filter(None, dsum))
        dsum = ' '.join(dsum)
        sum_arr.append(dsum)

        dsrc = re.split(r'\s', arr[2])
        dsrc = list(filter(None, dsrc))
        src_arr.append(dsrc)
        dsrc2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                   for wd in dsrc]
        src_idx.append(dsrc2id)
        dsrc2id = [vocab2id[wd] if wd in vocab2id else ext_vocab[wd]
                   for wd in dsrc]
        src_idx_ex.append(dsrc2id)
        dsrc2wt = [0.0 if wd in vocab2id else 1.0 for wd in dsrc]
        src_wt.append(dsrc2wt)
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

    return ext_id2oov, src_var, src_var_ex, src_arr, src_msk, sum_arr, ttl_arr


def process_data_app(data, vocab2id, src_lens):
    # build extended vocabulary
    ext_vocab = {}
    ext_id2oov = {}
    for wd in data["content_token"]:
        if wd not in vocab2id:
            ext_vocab[wd] = {}
    cnt = len(vocab2id)
    for wd in ext_vocab:
        ext_vocab[wd] = cnt
        ext_id2oov[cnt] = wd
        cnt += 1

    src_arr = []
    src_idx = []
    src_idx_ex = []
    src_wt = []

    dart = data["content_token"]
    src_arr.append(dart)
    dart2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
               for wd in dart]
    src_idx.append(dart2id)
    dart2id = [vocab2id[wd] if wd in vocab2id else ext_vocab[wd]
               for wd in dart]
    src_idx_ex.append(dart2id)
    dart2wt = [0.0 if wd in vocab2id else 1.0 for wd in dart]
    src_wt.append(dart2wt)

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

    return ext_id2oov, src_var, src_var_ex, src_arr, src_msk
