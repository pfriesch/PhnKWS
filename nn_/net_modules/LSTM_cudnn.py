import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class LSTM(nn.Module):

    def __init__(self, inp_dim, hidden_size, num_layers, bias, batch_first, dropout, bidirectional):
        super(LSTM, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bias=bias, dropout=dropout, batch_first=batch_first, bidirectional=self.bidirectional)

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if isinstance(x, PackedSequence):
            batch_size = x.batch_sizes[0]
        else:
            batch_size = x.shape[1]

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        output, (hn, cn) = self.lstm(x, (h0, c0))

        return output
