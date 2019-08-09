'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class natsEncoder(torch.nn.Module):
    '''
    RNN encoder for nats
    '''

    def __init__(self, emb_dim, hidden_size,
                 rnn_network, device=torch.device("cpu")):
        super(natsEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_network = rnn_network
        self.device = device

        if rnn_network == 'lstm':
            self.encoder = torch.nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True).to(device)
        elif rnn_network == 'gru':
            self.encoder = torch.nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True).to(device)

    def forward(self, input_):
        '''
        RNN encoder for nats
        '''
        batch_size = input_.size(0)

        h0_encoder = Variable(torch.zeros(
            2, batch_size, self.hidden_size)).to(self.device)
        if self.rnn_network == 'lstm':
            c0_encoder = Variable(torch.zeros(
                2, batch_size, self.hidden_size)).to(self.device)
            # encoding
            encoder_hy, (src_h_t, src_c_t) = self.encoder(
                input_, (h0_encoder, c0_encoder))

            return encoder_hy, (src_h_t, src_c_t)

        elif self.rnn_network == 'gru':
            # encoding
            encoder_hy, src_h_t = self.encoder(
                input_, h0_encoder)

            return encoder_hy, src_h_t
