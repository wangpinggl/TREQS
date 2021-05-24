'''
@author Ping Wang and Tian Shi
Please contact ping@vt.edu or tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from seq2sql.model_seq2seq_base import modelSeq2SeqBase
from LeafNATS.data.seq2sql.process_batch_cqa_v1 import process_batch

from LeafNATS.modules.embedding.nats_embedding import natsEmbedding
from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN
from LeafNATS.modules.encoder2decoder.nats_encoder2decoder import natsEncoder2Decoder
from LeafNATS.modules.attention.nats_attention_encoder import AttentionEncoder
from LeafNATS.modules.attention.nats_attention_decoder import AttentionDecoder

# from LeafNATS.utils.utils import *

class modelABS(modelSeq2SeqBase):
    
    def __init__(self, args):
        super().__init__(args=args)
        
    def build_scheduler(self, optimizer):
        '''
        Schedule Learning Rate
        '''
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=self.args.step_size, 
            gamma=self.args.step_decay)
        
        return scheduler
        
    def build_batch(self, batch_id):
        '''
        get batch data
        '''
        output = process_batch(
            batch_id=batch_id, 
            path_=os.path.join('..', 'nats_results'), 
            fkey_=self.args.task, 
            batch_size=self.args.batch_size, 
            vocab2id=self.batch_data['vocab2id'], 
            max_lens=[self.args.src_seq_len, self.args.trg_seq_len])
            
        self.batch_data['ext_id2oov'] = output['ext_id2oov']
        self.batch_data['src_var'] = output['src_var'].to(self.args.device)
        self.batch_data['batch_size'] = self.batch_data['src_var'].size(0)
        self.batch_data['src_seq_len'] = self.batch_data['src_var'].size(1)
        self.batch_data['src_mask_pad'] = output['src_mask_pad'].to(self.args.device)
        
        if self.args.task == 'train' or self.args.task == 'validate':
            self.batch_data['trg_input'] = output['trg_input_var'].to(self.args.device)
            # different from seq2seq models.
            self.batch_data['trg_output'] = output['trg_output_var'].to(self.args.device)
            self.batch_data['trg_seq_len'] = self.batch_data['trg_input'].size(1)
        else:
            self.batch_data['src_mask_unk'] = output['src_mask_unk'].to(self.args.device)
            self.batch_data['src_txt'] = output['src_txt']
            self.batch_data['trg_txt'] = output['trg_txt']
            self.batch_data['trg_seq_len'] = 1
        
    def build_models(self):
        '''
        build all models.
        in this model source and target share embeddings
        '''
        self.train_models['embedding'] = natsEmbedding(
            vocab_size = self.batch_data['vocab_size'],
            emb_dim = self.args.emb_dim,
            share_emb_weight = True
        ).to(self.args.device)
        
        self.train_models['encoder'] = EncoderRNN(
            self.args.emb_dim, self.args.src_hidden_dim,
            self.args.nLayers, 'lstm',
            device = self.args.device
        ).to(self.args.device)
        
        self.train_models['encoder2decoder'] = natsEncoder2Decoder(
            src_hidden_size = self.args.src_hidden_dim,
            trg_hidden_size = self.args.trg_hidden_dim,
            rnn_network = 'lstm',
            device = self.args.device
        ).to(self.args.device)
        
        self.train_models['decoderRNN'] = torch.nn.LSTMCell(
            self.args.emb_dim+self.args.trg_hidden_dim, 
            self.args.trg_hidden_dim
        ).to(self.args.device)
        
        self.train_models['attnEncoder'] = AttentionEncoder(
            self.args.src_hidden_dim,
            self.args.trg_hidden_dim,
            attn_method='luong_general',
            repetition='temporal'
        ).to(self.args.device)
        
        self.train_models['attnDecoder'] = AttentionDecoder(
            self.args.trg_hidden_dim,
            attn_method='luong_general'
        ).to(self.args.device)
        
        self.train_models['wrapDecoder'] = torch.nn.Linear(
            self.args.src_hidden_dim*2+self.args.trg_hidden_dim*2,
            self.args.trg_hidden_dim, bias=True
        ).to(self.args.device)
        
        self.train_models['genPrb'] = torch.nn.Linear(
            self.args.emb_dim+self.args.src_hidden_dim*2+self.args.trg_hidden_dim, 1
        ).to(self.args.device)
        
        # decoder to vocab
        self.train_models['decoder2proj'] = torch.nn.Linear(
            self.args.trg_hidden_dim, self.args.emb_dim, bias=False
        ).to(self.args.device)
                    
    def build_encoder(self):
        '''
        Encoder Pipeline
        self.pipe_data = {
            'encoder': {},
            'decoderA': {}}
            'decoderB': {'accu_attn': [], 'last_word': word}}
        '''
        src_emb = self.train_models['embedding'].get_embedding(
            self.batch_data['src_var'])
        src_enc, hidden_encoder = self.train_models['encoder'](src_emb)
        trg_hidden0 = self.train_models['encoder2decoder'](hidden_encoder)
        
        # set up pipe_data pass to decoder
        self.pipe_data['encoder'] = {}
        self.pipe_data['encoder']['src_emb'] = src_emb
        self.pipe_data['encoder']['src_enc'] = src_enc
        
        self.pipe_data['decoderB'] = {}
        self.pipe_data['decoderB']['hidden'] = trg_hidden0        
        self.pipe_data['decoderB']['h_attn'] = Variable(torch.zeros(
            self.batch_data['batch_size'], self.args.trg_hidden_dim
        )).to(self.args.device)
        self.pipe_data['decoderB']['past_attn'] = Variable(torch.ones(
            self.batch_data['batch_size'], self.batch_data['src_seq_len']
        )/float(self.batch_data['src_seq_len'])).to(self.args.device)
        self.pipe_data['decoderB']['past_dech'] = Variable(torch.zeros(
            1, 1)).to(self.args.device)
        self.pipe_data['decoderB']['accu_attn'] = []
        
        self.pipe_data['decoderFF'] = {}
        self.pipe_data['decoderFF']['h_attn'] = []
        self.pipe_data['decoderFF']['attn'] = []
        self.pipe_data['decoderFF']['genPrb'] = []
        # when training get target embedding at the same time.
        if self.args.task == 'train' or self.args.task == 'validate':
            trg_emb = self.train_models['embedding'].get_embedding(
                self.batch_data['trg_input'])
            self.pipe_data['decoderFF']['trg_seq_emb'] = trg_emb
        
    def build_decoder_one_step(self, k=0):
        '''
        Decoder one-step
        '''
        # embedding at current decoding step
        if self.args.task == 'train' or self.args.task == 'validate':
            self.pipe_data['decoderA'] = self.pipe_data['decoderB']
            word_emb = self.pipe_data['decoderFF']['trg_seq_emb'][:, k]
        else:
            word_emb = self.train_models['embedding'].get_embedding(
                self.pipe_data['decoderA']['last_word'])

        h_attn = self.pipe_data['decoderA']['h_attn']
        dec_input = torch.cat((word_emb, h_attn), 1)
        hidden = self.pipe_data['decoderA']['hidden']
        past_attn = self.pipe_data['decoderA']['past_attn']
        accu_attn = self.pipe_data['decoderA']['accu_attn']
        past_dech = self.pipe_data['decoderA']['past_dech']
        
        hidden = self.train_models['decoderRNN'](dec_input, hidden)
        ctx_enc, attn, attn_ee = self.train_models['attnEncoder'](
            hidden[0], self.pipe_data['encoder']['src_enc'], 
            past_attn, self.batch_data['src_mask_pad'])
        # temporal attention
        past_attn = past_attn + attn_ee
        # decoder attention
        if k == 0:
            ctx_dec = Variable(torch.zeros(
                self.batch_data['batch_size'], self.args.trg_hidden_dim
            )).to(self.args.device)
        else:
            ctx_dec, _ = self.train_models['attnDecoder'](
                hidden[0], past_dech)
        past_dech = past_dech.transpose(0, 1) # seqL*batch*hidden
        dec_idx = past_dech.size(0)
        if k == 0:
            past_dech = hidden[0].unsqueeze(0) # seqL*batch*hidden
            past_dech = past_dech.transpose(0, 1) # batch*seqL*hidden
        else:
            past_dech = past_dech.contiguous().view(
                -1, self.args.trg_hidden_dim) # seqL*batch**hidden
            past_dech = torch.cat((past_dech, hidden[0]), 0) # (seqL+1)*batch**hidden
            past_dech = past_dech.view(
                dec_idx+1, self.batch_data['batch_size'], self.args.trg_hidden_dim
            ) # (seqL+1)*batch*hidden
            past_dech = past_dech.transpose(0, 1) # batch*(seqL+1)*hidden
        # wrap up.
        h_attn = self.train_models['wrapDecoder'](torch.cat((ctx_enc, ctx_dec, hidden[0]), 1))
        # pointer generator
        pt_input = torch.cat((word_emb, hidden[0], ctx_enc), 1)
        genPrb = torch.sigmoid(self.train_models['genPrb'](pt_input))
        
        # setup piped_data
        self.pipe_data['decoderB'] = {}
        self.pipe_data['decoderB']['h_attn'] = h_attn
        self.pipe_data['decoderB']['past_attn'] = past_attn
        self.pipe_data['decoderB']['hidden'] = hidden
        self.pipe_data['decoderB']['past_dech'] = past_dech
        self.pipe_data['decoderB']['accu_attn'] = [a for a in accu_attn]
        self.pipe_data['decoderB']['accu_attn'].append(attn)
        
        if self.args.task == 'train' or self.args.task == 'validate':
            self.pipe_data['decoderFF']['h_attn'].append(h_attn)
            self.pipe_data['decoderFF']['attn'].append(attn)
            self.pipe_data['decoderFF']['genPrb'].append(genPrb)
            if k == self.batch_data['trg_seq_len']-1:
                self.pipe_data['decoderFF']['h_attn'] = \
                torch.cat(self.pipe_data['decoderFF']['h_attn'], 0).view(
                    self.batch_data['trg_seq_len'], 
                    self.batch_data['batch_size'], 
                    self.args.trg_hidden_dim).transpose(0,1)
                
                self.pipe_data['decoderFF']['attn'] = \
                torch.cat(self.pipe_data['decoderFF']['attn'], 0).view(
                    self.batch_data['trg_seq_len'], 
                    self.batch_data['batch_size'], 
                    self.args.src_seq_len).transpose(0,1)
                
                self.pipe_data['decoderFF']['genPrb'] = \
                torch.cat(self.pipe_data['decoderFF']['genPrb'], 0).view(
                    self.batch_data['trg_seq_len'], 
                    self.batch_data['batch_size']).transpose(0,1)
        else:
            self.pipe_data['decoderFF']['h_attn'] = h_attn
            self.pipe_data['decoderFF']['attn'] = attn.unsqueeze(0)
            self.pipe_data['decoderFF']['genPrb'] = genPrb

    def build_vocab_distribution(self):
        '''
        Data flow from input to output.
        '''
        trg_out = self.pipe_data['decoderFF']['h_attn']
        trg_out = self.train_models['decoder2proj'](trg_out)
        trg_out = self.train_models['embedding'].get_decode2vocab(trg_out)
        trg_out = trg_out.view(
            self.batch_data['batch_size'], self.batch_data['trg_seq_len'], -1)
        prb = torch.softmax(trg_out, dim=2)
        
        vocab_size = self.batch_data['vocab_size']
        batch_size = self.batch_data['batch_size']
        # trg_seq_len = self.batch_data['trg_seq_len']
        src_seq_len = self.batch_data['src_seq_len']
        
        # pointer-generator calculate index matrix
        pt_idx = Variable(torch.FloatTensor(torch.zeros(1, 1, 1))).to(self.args.device)
        pt_idx = pt_idx.repeat(batch_size, src_seq_len, vocab_size)
        pt_idx.scatter_(2, self.batch_data['src_var'].unsqueeze(2), 1.0)
               
        p_gen = self.pipe_data['decoderFF']['genPrb']
        attn_ = self.pipe_data['decoderFF']['attn']
        
        prb_output = p_gen.unsqueeze(2)*prb + \
                     (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)
                
        return prb_output + 1e-20
    
    def build_pipelines(self):
        '''
        Build pipeline from input to output.
        Output is loss.
        Input is word one-hot encoding.
        '''
        self.build_encoder()
        for k in range(self.args.trg_seq_len):
            self.build_decoder_one_step(k)
        prb = self.build_vocab_distribution()
        
        pad_mask = torch.ones(self.batch_data['vocab_size']).to(self.args.device)
        pad_mask[self.batch_data['vocab2id']['<pad>']] = 0
        self.loss_criterion = torch.nn.NLLLoss(pad_mask).to(self.args.device)
        
        prb = torch.log(prb)
        loss = self.loss_criterion(
            prb.view(-1, self.batch_data['vocab_size']),
            self.batch_data['trg_output'].view(-1))
        
        return loss
    
    
