from distutils.util import strtobool

import torch
import torch.nn as nn


class GRU_cudnn(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = options['hidden_size']
        self.num_layers = options['num_layers']
        self.bias = options['bias']
        self.batch_first = options['batch_first']
        self.dropout = options['dropout']
        self.bidirectional = options['bidirectional']

        self.gru = nn.ModuleList([nn.GRU(self.input_dim, self.hidden_size, self.num_layers,
                                         bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)])

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()

        output, hn = self.gru[0](x, h0)

        return output
