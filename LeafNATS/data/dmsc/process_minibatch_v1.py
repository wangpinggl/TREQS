'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch
from torch.autograd import Variable


def process_minibatch(input_, vocab2id, max_lens):
    '''
    Process the minibatch.

    ID Meta<sec>Features<sec>ratings<sec>review
    e.g., 0<sec>0<sec>1 3 1 4<sec>review
    '''
    len_review = []
    review_arr = []
    rating_arr = []
    feat_arr = []
    for line in input_:
        arr = re.split('<sec>', line[:-1].lower())

        tmp_rate = re.split(r'\s', arr[-2])
        rating_arr.append([int(round(float(itm)))-1 for itm in tmp_rate[:-1]])

        tmp_feat = [float(wd) for wd in re.split(r'\s', arr[1])]
        feat_arr.append(tmp_feat)

        review = re.split(r'\s|<s>|</s>', arr[-1])
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
    feat_var = Variable(torch.FloatTensor(feat_arr))

    weight_mask = Variable(torch.FloatTensor(review_arr))
    weight_mask[weight_mask != 1.0] = 0.0
    weight_mask = 1.0 - weight_mask

    return review_var, weight_mask, rating_var, feat_var
