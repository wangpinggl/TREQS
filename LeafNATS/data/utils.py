'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import os
import random
import re
import shutil

import numpy as np
import torch
from torch.autograd import Variable


def construct_vocab(file_,
                    max_size=200000,
                    mincount=5):
    '''
    Construct vocabulary
    token<sec>count
    or
    token count
    '''
    vocab2id = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}
    id2vocab = {2: '<s>', 3: '</s>', 1: '<pad>', 0: '<unk>', 4: '<stop>'}
    word_pad = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}

    cnt = len(vocab2id)
    with open(file_, 'r') as fp:
        for line in fp:
            arr = re.split(' ', line[:-1])
            if len(arr) == 1:
                arr = re.split('<sec>', line[:-1])
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            if int(arr[1]) >= mincount:
                vocab2id[arr[0]] = cnt
                id2vocab[cnt] = arr[0]
                cnt += 1
            if len(vocab2id) == max_size:
                break

    return vocab2id, id2vocab


def load_vocab_pretrain(file_pretrain_vocab,
                        file_pretrain_vec):
    '''
    Load pretrained embedding
    token<sec>count
    or
    token count
    '''
    vocab2id = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}
    id2vocab = {2: '<s>', 3: '</s>', 1: '<pad>', 0: '<unk>', 4: '<stop>'}
    word_pad = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}

    pad_cnt = len(vocab2id)
    cnt = len(vocab2id)
    with open(file_pretrain_vocab, 'r') as fp:
        for line in fp:
            arr = re.split(' ', line[:-1])
            if len(arr) == 1:
                arr = re.split('<sec>', line[:-1])
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            vocab2id[arr[0]] = cnt
            id2vocab[cnt] = arr[0]
            cnt += 1

    pretrain_vec = np.load(file_pretrain_vec)
    pad_vec = np.zeros([pad_cnt, pretrain_vec.shape[1]])
    pretrain_vec = np.vstack((pad_vec, pretrain_vec))

    return vocab2id, id2vocab, pretrain_vec


def construct_pos_vocab(file_):
    '''
    Construct vocabulary for part of speech tagging
    tag count
    or
    tag<sec>count
    '''
    vocab2id = {'<pad>': 0}
    id2vocab = {0: '<pad>'}
    word_pad = {'<pad>': 0}

    cnt = len(vocab2id)
    with open(file_, 'r') as fp:
        for line in fp:
            arr = re.split(' ', line[:-1])
            if len(arr) == 1:
                arr = re.split('<sec>', line[:-1])
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            vocab2id[arr[0]] = cnt
            id2vocab[cnt] = arr[0]
            cnt += 1

    return vocab2id, id2vocab


def construct_char_vocab(file_):
    '''
    Construct vocabulary for characters
    char<sec>count
    or
    char count
    '''
    vocab2id = {'<pad>': 0}
    id2vocab = {0: '<pad>'}
    word_pad = {'<pad>': 0}

    cnt = len(vocab2id)
    with open(file_, 'r') as fp:
        for line in fp:
            arr = re.split(' ', line[:-1])
            if len(arr) == 1:
                arr = re.split('<sec>', line[:-1])
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            vocab2id[arr[0]] = cnt
            id2vocab[cnt] = arr[0]
            cnt += 1

    return vocab2id, id2vocab


def create_batch_file(path_data,  # path to data dir
                      path_work,  # path to work dir
                      is_shuffle,
                      fkey_,  # train dev test
                      file_,  # input file
                      batch_size,
                      is_lower=True):
    '''
    Users cannot rewrite this function, unless they want to rewrite the engine.

    Split the corpus into batches. Data store in hard drive.
    Used when you have a very large corpus.
    advantage: Don't worry about the memeory.
    disadvantage: Takes some time to split the batches.
    '''
    file_name = os.path.join(path_data, file_)
    folder = os.path.join(path_work, 'batch_'+fkey_+'_'+str(batch_size))

    try:
        shutil.rmtree(folder)
        os.mkdir(folder)
    except:
        os.mkdir(folder)

    corpus_arr = []
    fp = open(file_name, 'r', encoding="iso-8859-1")
    for line in fp:
        if is_lower:
            line = line.lower()
        corpus_arr.append(line)
    fp.close()
    if is_shuffle:
        random.shuffle(corpus_arr)

    cnt = 0
    for itm in corpus_arr:
        try:
            arr.append(itm)
        except:
            arr = [itm]
        if len(arr) == batch_size:
            fout = open(os.path.join(folder, str(cnt)), 'w')
            for sen in arr:
                fout.write(sen)
            fout.close()
            arr = []
            cnt += 1

    if len(arr) > 0:
        fout = open(os.path.join(folder, str(cnt)), 'w')
        for sen in arr:
            fout.write(sen)
        fout.close()
        arr = []
        cnt += 1

    return cnt


def create_batch_memory(path_,  # path to data
                        file_,  # input file
                        is_shuffle,
                        batch_size,
                        is_lower=True):
    '''
    Users cannot rewrite this function, unless they want to rewrite the engine.

    used when the data is relatively small.
    This will store data in memeory.
    Advantage: Fast and easy to handle.
    '''

    file_name = os.path.join(path_, file_)

    corpus_arr = []
    fp = open(file_name, 'r', encoding="iso-8859-1")
    for line in fp:
        if is_lower:
            line = line.lower()
        corpus_arr.append(line)
    fp.close()
    if is_shuffle:
        random.shuffle(corpus_arr)

    data_split = []
    for itm in corpus_arr:
        try:
            arr.append(itm)
        except:
            arr = [itm]
        if len(arr) == batch_size:
            data_split.append(arr)
            arr = []

    if len(arr) > 0:
        data_split.append(arr)
        arr = []

    return data_split
