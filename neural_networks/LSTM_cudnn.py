from distutils.util import strtobool

import torch
import torch.nn as nn


class LSTM_cudnn(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options['hidden_size'])
        self.num_layers = int(options['num_layers'])
        self.bias = strtobool(options['bias'])
        self.batch_first = strtobool(options['batch_first'])
        self.dropout = float(options['dropout'])
        self.bidirectional = strtobool(options['bidirectional'])

        self.lstm = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                           bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)])

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        output, (hn, cn) = self.lstm[0](x, (h0, c0))

        return output
