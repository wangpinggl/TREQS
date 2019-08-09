'''
@author Ping Wang and Tian Shi
Please contact ping@vt.edu or tshi@vt.edu
'''
import os
import time
import copy
import numpy as np

import torch
from torch.autograd import Variable

from LeafNATS.engines.end2end_large import End2EndBase
from LeafNATS.data.utils import construct_vocab
from LeafNATS.modules.decoding.word_copy import word_copy

class modelSeq2SeqBase(End2EndBase):
    '''
    Seq2sql based on seq2seq models.
    ''' 
    def __init__(self, args):
        super().__init__(args=args)
        
        self.pipe_data = {} # for pipe line
        self.beam_data = [] # for beam search
        
    def build_vocabulary(self):
        '''
        build vocabulary
        '''
        vocab2id, id2vocab = construct_vocab(
            file_=os.path.join(self.args.data_dir, self.args.file_vocab),
            max_size=self.args.max_vocab_size,
            mincount=self.args.word_minfreq)
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))
        
    def build_optimizer(self, params):
        '''
        Build model optimizer
        '''
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
                
        return optimizer
    
    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        for model_name in self.base_models:
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))
    
    def init_train_model_params(self):
        '''
        Initialize Train Model Parameters.
        For testing and visulization.
        '''
        for model_name in self.train_models:
            fl_ = os.path.join(
                self.args.train_model_dir, 
                model_name+'_'+str(self.args.best_model)+'.model')
            self.train_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))
    
    def build_encoder(self):
        '''
        Encoder, Encoder2Decoder, etc..
        '''
        raise NotImplementedError
                
    def build_decoder_one_step(self):
        '''
        Just decode for one step.
        '''
        raise NotImplementedError
        
    def build_vocab_distribution(self):
        '''
        Calculate Vocabulary Distribution
        '''
        raise NotImplementedError
    
    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        Output is loss.
        Input is word one-hot encoding.
        '''
        self.build_encoder()
        for k in range(self.args.trg_seq_len):
            self.build_decoder_one_step(k)
        prob = self.build_vocab_distribution()
        
        prob = torch.log(prob)
        loss = self.loss_criterion(
            prob.view(-1, self.batch_data['vocab_size']),
            self.batch_data['trg_output'].view(-1))
        
        return loss
    
    def beam_search(self):
        '''
        Light Weight Beam Search Algorithm.
        '''
        self.build_encoder()
        lastwd = Variable(torch.LongTensor([
            self.batch_data['vocab2id']['select']])).to(self.args.device)
        self.beam_data = [[[lastwd], 0.0, self.pipe_data['decoderB']]]
                
        for k in range(self.args.trg_seq_len):
            beam_tmp = []
            for j in range(len(self.beam_data)):
                lastwd = self.beam_data[j][0][-1]
                if lastwd == self.batch_data['vocab2id']['<stop>']:
                    beam_tmp.append(self.beam_data[j])
                    continue
                self.pipe_data['decoderA'] = self.beam_data[j][2]
                if lastwd >= len(self.batch_data['vocab2id']):
                    lastwd = Variable(torch.LongTensor([
                        self.batch_data['vocab2id']['<unk>']]
                    )).to(self.args.device)
                self.pipe_data['decoderA']['last_word'] = lastwd
                self.build_decoder_one_step(k)
                prob = self.build_vocab_distribution()
                prob = torch.log(prob)
                
                score, wd_list = prob.data.topk(self.args.beam_size)
                score = score.squeeze()
                wd_list = wd_list.squeeze()
                seq_base = self.beam_data[j][0]
                prob_base = self.beam_data[j][1]*float(len(seq_base))
                for i in range(self.args.beam_size):
                    beam_tmp.append([
                        seq_base+[wd_list[i].view(1)], 
                        (prob_base+score[i])/float(len(seq_base)+1), 
                        self.pipe_data['decoderB']])
            beam_tmp = sorted(beam_tmp, key=lambda pb: pb[1])[::-1]
            self.beam_data = beam_tmp[:self.args.beam_size]
    
    def word_copy(self):
        '''
        copy words from source document.
        '''
        myseq = torch.cat(self.beam_data[0][0], 0)
        myattn = torch.cat(self.beam_data[0][-1]['accu_attn'], 0)
        myattn = myattn*self.batch_data['src_mask_unk']
        beam_copy = myattn.topk(1, dim=1)[1].squeeze(-1)
        wdidx = beam_copy.data.cpu().numpy()
        out_txt = []
        myseq = torch.cat(self.beam_data[0][0], 0)
        myseq = myseq.data.cpu().numpy().tolist()
        gen_txt = [self.batch_data['id2vocab'][wd] 
                    if wd in self.batch_data['id2vocab'] 
                    else self.batch_data['ext_id2oov'][wd] 
                    for wd in myseq]
        for j in range(len(gen_txt)):
            if gen_txt[j] == '<unk>':
                gen_txt[j] = self.batch_data['src_txt'][0][wdidx[j]]
        out_txt.append(' '.join(gen_txt))

        return out_txt
    
    def test_worker(self):
        '''
        For the beam search in testing.
        '''
        self.beam_search()
        try:
            myseq = self.word_copy()
        except:
            print('Running without manually word copying.')
            myseq = torch.cat(self.beam_data[0][0], 0)
            myseq = myseq.data.cpu().numpy().tolist()
            myseq = [self.batch_data['id2vocab'][idx] for idx in myseq]

        self.test_data['sql_gold'] = ' '.join(self.batch_data['trg_txt'][0])
        self.test_data['sql_pred'] = ' '.join(myseq)
