'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class ReadAgainEncoder(torch.nn.Module):

    def __init__(self, emb_dim,
                 hidden_size, rnn_network,
                 device=torch.device("cpu")):
        '''
        Read-again encoder
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_network = rnn_network
        self.device = device

        if rnn_network == 'lstm':
            self.encoder1 = torch.nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True).to(device)
            self.encoder2 = torch.nn.LSTM(
                input_size=emb_dim+hidden_size*2+hidden_size*2,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True).to(device)
        elif rnn_network == 'gru':
            self.encoder1 = torch.nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True).to(device)
            self.encoder2 = torch.nn.GRU(
                input_size=emb_dim+hidden_size*2+hidden_size*2,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True).to(device)

    def forward(self, input_):
        '''
        get encoding
        '''
        batch_size = input_.size(0)

        h0_encoder1 = Variable(torch.zeros(
            2, batch_size, self.hidden_size)).to(self.device)
        h0_encoder2 = Variable(torch.zeros(
            2, batch_size, self.hidden_size)).to(self.device)
        if self.rnn_network == 'lstm':
            c0_encoder1 = Variable(torch.zeros(
                2, batch_size, self.hidden_size)).to(self.device)
            c0_encoder2 = Variable(torch.zeros(
                2, batch_size, self.hidden_size)).to(self.device)
            # encoding
            hy_encoder1, (ht_encoder1, ct_encoder1) = self.encoder1(
                input_, (h0_encoder, c0_encoder))
            ht_encoder1 = ht_encoder1.transpose(
                0, 1).contiguous().view(batch_size, -1)
            ht_encoder1 = ht_encoder1.unsqueeze(
                1).repeat(1, hy_encoder1.size(1), 1)
            input_encoder2 = torch.cat((input_, hy_encoder1, ht_encoder1), 2)
            hy_encoder2, (ht_encoder2, ct_encoder2) = self.encoder2(
                input_encoder2, (h0_encoder2, c0_encoder2))

            return hy_encoder2, (ht_encoder2, ct_encoder2)

        elif self.rnn_network == 'gru':
            # encoding
            hy_encoder1, ht_encoder1 = self.encoder1(input_, h0_encoder1)
            ht_encoder1 = ht_encoder1.transpose(
                0, 1).contiguous().view(batch_size, -1)
            ht_encoder1 = ht_encoder1.unsqueeze(
                1).repeat(1, hy_encoder1.size(1), 1)
            input_encoder2 = torch.cat((input_, hy_encoder1, ht_encoder1), 2)
            hy_encoder2, ht_encoder2 = self.encoder2(
                input_encoder2, h0_encoder2)

            return hy_encoder2, ht_encoder2
