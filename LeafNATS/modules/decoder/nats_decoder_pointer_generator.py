'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable

from LeafNATS.modules.attention.nats_attention_decoder import AttentionDecoder
from LeafNATS.modules.attention.nats_attention_encoder import AttentionEncoder


class PointerGeneratorDecoder(torch.nn.Module):
    def __init__(self, input_size,
                 src_hidden_size, trg_hidden_size,
                 attn_method, repetition,
                 pointer_net,  attn_decoder, rnn_network,
                 device=torch.device("cpu")):
        '''
        LSTM/GRU decoder
        Seq2Seq attention decoder
        pointer-generator network decoder

        input_size, # input vector size
        src_hidden_size, # source side hidden size
        trg_hidden_size, # target side hidden size
        attn_method, # alignment methods
        repetition, # approaches handle repetition
        pointer_net, # turn on pointer network?
        attn_decoder, # turn on attention decoder?
        '''
        super(PointerGeneratorDecoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.src_hidden_size = src_hidden_size
        self.trg_hidden_size = trg_hidden_size
        self.attn_method = attn_method.lower()
        self.repetition = repetition
        self.pointer_net = pointer_net
        self.attn_decoder = attn_decoder
        self.rnn_network = rnn_network
        self.device = device

        if rnn_network == 'lstm':
            self.rnn_ = torch.nn.LSTMCell(
                input_size+trg_hidden_size,
                trg_hidden_size).to(device)
        else:
            self.rnn_ = torch.nn.GRUCell(
                input_size+trg_hidden_size,
                trg_hidden_size).to(device)

        self.encoder_attn_layer = AttentionEncoder(
            src_hidden_size=src_hidden_size,
            trg_hidden_size=trg_hidden_size,
            attn_method=attn_method,
            repetition=repetition).to(device)
        # intra-decoder
        if self.attn_decoder:
            self.decoder_attn_layer = AttentionDecoder(
                hidden_size=trg_hidden_size,
                attn_method=attn_method).to(device)
            self.attn_out = torch.nn.Linear(
                src_hidden_size*2+trg_hidden_size*2,
                trg_hidden_size).to(device)
        else:
            self.attn_out = torch.nn.Linear(
                src_hidden_size*2+trg_hidden_size,
                trg_hidden_size).to(device)
        # pointer generator network
        if self.pointer_net:
            if self.attn_decoder:
                self.pt_out = torch.nn.Linear(
                    input_size+src_hidden_size*2+trg_hidden_size*2, 1).to(device)
            else:
                self.pt_out = torch.nn.Linear(
                    input_size+src_hidden_size*2+trg_hidden_size, 1).to(device)

    def forward(
        self,
        # index of current decoding step (used in testing, set 0 for training)
        idx,
        input_,  # input embedding
        hidden_,  # hidden and cell states
        h_attn,  # hidden output
        encoder_hy,  # encoder hidden states
        past_attn,  # previous attention
        p_gen,  # pointer generator soft-switch
        past_dehy  # previous decoder hidden states
    ):

        input_ = input_.transpose(0, 1)
        batch_size = input_.size(1)

        output_ = []
        out_attn = []

        loss_cv = Variable(torch.zeros(1)).to(self.device)
        batch_size = input_.size(1)
        for k in range(input_.size(0)):
            x_input = torch.cat((input_[k], h_attn), 1)
            hidden_ = self.rnn_(x_input, hidden_)
            if self.rnn_network == 'lstm':
                hhh_ = hidden_[0]
            else:
                hhh_ = hidden_
            # attention encoder
            c_encoder, attn, attn_ee = self.encoder_attn_layer(
                hhh_, encoder_hy, past_attn)
            # attention decoder
            if self.attn_decoder:
                if k + idx == 0:
                    c_decoder = Variable(torch.zeros(
                        batch_size, self.trg_hidden_size)).to(self.device)
                else:
                    c_decoder, _ = self.decoder_attn_layer(
                        hhh_, past_dehy)
                past_dehy = past_dehy.transpose(0, 1)  # seqL*batch*hidden
                de_idx = past_dehy.size(0)
                if k + idx == 0:
                    past_dehy = hhh_.unsqueeze(0)  # seqL*batch*hidden
                    past_dehy = past_dehy.transpose(0, 1)  # batch*seqL*hidden
                else:
                    past_dehy = past_dehy.contiguous().view(-1, self.trg_hidden_size)  # seqL*batch**hidden
                    # (seqL+1)*batch**hidden
                    past_dehy = torch.cat((past_dehy, hhh_), 0)
                    past_dehy = past_dehy.view(
                        de_idx+1, batch_size, self.trg_hidden_size)  # (seqL+1)*batch*hidden
                    past_dehy = past_dehy.transpose(
                        0, 1)  # batch*(seqL+1)*hidden
                h_attn = self.attn_out(
                    torch.cat((c_encoder, c_decoder, hhh_), 1))
            else:
                h_attn = self.attn_out(torch.cat((c_encoder, hhh_), 1))
            # repetition
            if self.repetition == 'asee_train':
                lscv = torch.cat(
                    (past_attn.unsqueeze(2), attn.unsqueeze(2)), 2)
                lscv = lscv.min(dim=2)[0]
                try:
                    loss_cv = loss_cv + torch.mean(lscv)
                except:
                    loss_cv = torch.mean(lscv)
            if self.repetition[:4] == 'asee':
                past_attn = past_attn + attn
            if self.repetition == 'temporal':
                if k + idx == 0:
                    past_attn = past_attn*0.0
                past_attn = past_attn + attn_ee
            # output
            output_.append(h_attn)
            out_attn.append(attn)
            # pointer
            if self.pointer_net:
                if self.attn_decoder:
                    pt_input = torch.cat(
                        (input_[k], hhh_, c_encoder, c_decoder), 1)
                else:
                    pt_input = torch.cat((input_[k], hhh_, c_encoder), 1)
                p_gen[:, k] = torch.sigmoid(self.pt_out(pt_input).squeeze(1))

        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, batch_size, hidden_size)
        out_attn = torch.cat(out_attn, 0).view(
            len_seq, attn.size(0), attn.size(1))

        output_ = output_.transpose(0, 1)

        return output_, hidden_, h_attn, out_attn, past_attn, p_gen, past_dehy, loss_cv
