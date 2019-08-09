'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class EncoderRNN(torch.nn.Module):

    def __init__(self, emb_dim,
                 hidden_size, nLayers,
                 rnn_network, bidirectional=True,
                 device=torch.device("cpu")):
        '''
        RNN encoder
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_network = rnn_network
        self.nLayers = nLayers
        self.device = device
        self.bidirectional = bidirectional

        if rnn_network == 'lstm':
            self.encoder = torch.nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=nLayers,
                batch_first=True,
                bidirectional=bidirectional
            ).to(device)
        elif rnn_network == 'gru':
            self.encoder = torch.nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=nLayers,
                batch_first=True,
                bidirectional=bidirectional
            ).to(device)

    def forward(self, input_):
        '''
        get encoding
        '''
        n_dk = 1
        if self.bidirectional:
            n_dk = 2
        batch_size = input_.size(0)

        h0_encoder = Variable(torch.zeros(
            n_dk*self.nLayers, batch_size, self.hidden_size)).to(self.device)
        if self.rnn_network == 'lstm':
            c0_encoder = Variable(torch.zeros(
                n_dk*self.nLayers, batch_size, self.hidden_size)).to(self.device)
            # encoding
            hy_encoder, (ht_encoder, ct_encoder) = self.encoder(
                input_, (h0_encoder, c0_encoder))

            return hy_encoder, (ht_encoder, ct_encoder)

        elif self.rnn_network == 'gru':
            # encoding
            hy_encoder, ht_encoder = self.encoder(input_, h0_encoder)

            return hy_encoder, ht_encoder
