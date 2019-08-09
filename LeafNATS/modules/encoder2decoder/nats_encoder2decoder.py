'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class natsEncoder2Decoder(torch.nn.Module):

    def __init__(self, src_hidden_size,
                 trg_hidden_size, rnn_network,
                 device=torch.device("cpu")):
        '''
        encoder rnn 2 decoder rnn.
        '''
        super(natsEncoder2Decoder, self).__init__()
        self.rnn_network = rnn_network

        self.encoder2decoder = torch.nn.Linear(
            2*src_hidden_size, trg_hidden_size)
        if rnn_network == 'lstm':
            self.encoder2decoder_c = torch.nn.Linear(
                2*src_hidden_size, trg_hidden_size)

    def forward(self, hidden_encoder):
        if self.rnn_network == 'lstm':
            (src_h_t, src_c_t) = hidden_encoder
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

            decoder_h0 = torch.tanh(self.encoder2decoder(h_t))
            decoder_c0 = torch.tanh(self.encoder2decoder_c(c_t))

            return (decoder_h0, decoder_c0)
        else:
            src_h_t = hidden_encoder
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)

            decoder_h0 = torch.tanh(self.encoder2decoder(h_t))

            return decoder_h0
