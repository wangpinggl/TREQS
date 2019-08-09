'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import re

import torch
from torch.autograd import Variable

def process_minibatch(input_, 
                      vocab2id, 
                      vocab2id_char, 
                      vocab2id_pos, 
                      premise_max_lens, 
                      hypothesis_max_lens):
    '''
    Process the minibatch.
    {
        'premise': content, 
        'hypothesis': content, 
        'gold_label': 1, 
        'premise_pos': content, 
        'hypothesis_pos': content
    }
    '''
    len_premise = []
    len_hypothe = []
    premise_arr = []
    hypothe_arr = []
    prem_pos_arr = []
    hypo_pos_arr = []
    prem_char_arr = []
    hypo_char_arr = []
    label_arr = []
    for line in input_:
        data = json.loads(line)
        
        label_arr.append(data['gold_label']+1)
                
        premise = data['premise']
        len_premise.append(len(premise))
        premise2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>'] for wd in premise]
        premise_arr.append(premise2id)
        
        hypothe = data['hypothesis']
        len_hypothe.append(len(hypothe))
        hypothe2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>'] for wd in hypothe]
        hypothe_arr.append(hypothe2id)
        
        prem_char = [list(map(lambda x: vocab2id_char[x], list(wd))) for wd in premise]
        prem_char = [itm[:16] + [0]*(16-min(16, len(itm))) for itm in prem_char]
        prem_char_arr.append(prem_char)
        
        hypo_char = [list(map(lambda x: vocab2id_char[x], list(wd))) for wd in hypothe]
        hypo_char = [itm[:16] + [0]*(16-min(16, len(itm))) for itm in hypo_char]
        hypo_char_arr.append(hypo_char)
        
        prem_pos = data['premise_pos']
        prem_pos2id = list(map(lambda x: vocab2id_pos[x], prem_pos))
        prem_pos_arr.append(prem_pos2id)
        
        hypo_pos = data['hypothesis_pos']
        hypo_pos2id = list(map(lambda x: vocab2id_pos[x], hypo_pos))
        hypo_pos_arr.append(hypo_pos2id)
                
    premise_lens = min(premise_max_lens, max(len_premise))
    hypothe_lens = min(hypothesis_max_lens, max(len_hypothe))
        
    premise_arr = [itm[:premise_lens] for itm in premise_arr]
    premise_arr = [itm + [vocab2id['<pad>']]*(premise_lens-len(itm)) for itm in premise_arr]
    premise_var = Variable(torch.LongTensor(premise_arr))
    
    hypothe_arr = [itm[:hypothe_lens] for itm in hypothe_arr]
    hypothe_arr = [itm + [vocab2id['<pad>']]*(hypothe_lens-len(itm)) for itm in hypothe_arr]
    hypothe_var = Variable(torch.LongTensor(hypothe_arr))
    
    prem_char_arr = [itm[:premise_lens] for itm in prem_char_arr]
    prem_char_arr = [itm + [[0]*16]*(premise_lens-len(itm)) for itm in prem_char_arr]
    prem_char_var = Variable(torch.LongTensor(prem_char_arr))
    
    hypo_char_arr = [itm[:hypothe_lens] for itm in hypo_char_arr]
    hypo_char_arr = [itm + [[0]*16]*(hypothe_lens-len(itm)) for itm in hypo_char_arr]
    hypo_char_var = Variable(torch.LongTensor(hypo_char_arr))
        
    prem_pos_arr = [itm[:premise_lens] for itm in prem_pos_arr]
    prem_pos_arr = [itm + [vocab2id_pos['<pad>']]*(premise_lens-len(itm)) for itm in prem_pos_arr]
    prem_pos_var = Variable(torch.LongTensor(prem_pos_arr))
    
    hypo_pos_arr = [itm[:hypothe_lens] for itm in hypo_pos_arr]
    hypo_pos_arr = [itm + [vocab2id_pos['<pad>']]*(hypothe_lens-len(itm)) for itm in hypo_pos_arr]
    hypo_pos_var = Variable(torch.LongTensor(hypo_pos_arr))
    
    label_var = Variable(torch.LongTensor(label_arr))
    
    premise_mask = Variable(torch.FloatTensor(premise_arr))
    premise_mask[premise_mask!=1.0] = 0.0
    premise_mask = 1.0 - premise_mask
    
    hypothe_mask = Variable(torch.FloatTensor(hypothe_arr))
    hypothe_mask[hypothe_mask!=1.0] = 0.0
    hypothe_mask = 1.0 - hypothe_mask
    
    return premise_var, hypothe_var, prem_char_var, hypo_char_var, \
           prem_pos_var, hypo_pos_var, premise_mask, hypothe_mask, label_var
