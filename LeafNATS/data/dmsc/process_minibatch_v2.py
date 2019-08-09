'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch
from torch.autograd import Variable


def process_minibatch(input_, vocab2id, max_lens):
    '''
    Process the minibatch for beeradvocate and tripadvisor datasets
    The data format
    [0 overall] 1 3 1 4\t\t\tSOMETHING\t\t\treview
    For review, sentences are seperated by <ssssss>.
    '''
    len_review = []
    review_arr = []
    rating_arr = []
    for line in input_:
        arr = re.split('\t\t\t', line[:-1].lower())

        tmp_rate = re.split(r'\s', arr[0])[1:]
        tmp_rate = list(filter(None, tmp_rate))
        rating_arr.append([int(rt) for rt in tmp_rate])

        review = re.split(r'\s|<ssssss>', arr[-1])
        review = list(filter(None, review))
        len_review.append(len(review))

        review2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                     for wd in review]
        review_arr.append(review2id)

    review_lens = min(max_lens, max(len_review))

    review_arr = [itm[:review_lens] for itm in review_arr]
    review_arr = [itm + [vocab2id['<pad>']] *
                  (review_lens-len(itm)) for itm in review_arr]

    review_var = Variable(torch.LongTensor(review_arr))
    rating_var = Variable(torch.LongTensor(rating_arr))
    rating_var -= 1
    rating_var[rating_var < 0] = -1

    weight_mask = Variable(torch.FloatTensor(review_arr))
    weight_mask[weight_mask != float(vocab2id['<pad>'])] = -1.0
    weight_mask[weight_mask == float(vocab2id['<pad>'])] = 0.0
    weight_mask = -weight_mask

    return review_var, weight_mask, rating_var
