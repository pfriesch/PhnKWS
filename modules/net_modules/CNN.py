import torch.nn.functional as F
import torch.nn as nn

from modules.net_modules.utils import act_fun, LayerNorm


class CNN(nn.Module):

    def __init__(self, input_dim,
                 N_filters, len_filters,
                 max_pool_len,
                 use_laynorm,
                 use_batchnorm,
                 use_laynorm_inp,
                 use_batchnorm_inp,
                 activation,
                 dropout):
        super(CNN, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.use_laynorm = use_laynorm
        self.use_laynorm_inp = use_laynorm_inp
        self.use_batchnorm_inp = use_batchnorm_inp

        self.N_lay = len(N_filters)
        self.conv = nn.ModuleList([])
        self.batch_norm = nn.ModuleList([])
        self.layer_norm = nn.ModuleList([])
        self.activations = nn.ModuleList([])
        self.dropout = nn.ModuleList([])

        if self.use_laynorm_inp:
            self.layer_norm0 = LayerNorm(input_dim)

        if self.use_batchnorm_inp:
            self.batch_norm0 = nn.BatchNorm1d([input_dim], momentum=0.05)

        current_input = input_dim

        for i in range(self.N_lay):

            N_filters = int(N_filters[i])
            len_filters = int(len_filters[i])

            # dropout
            self.dropout.append(nn.Dropout(p=dropout[i]))

            # activation
            self.activations.append(act_fun(activation[i]))

            # layer norm initialization
            self.layer_norm.append(
                LayerNorm([N_filters, int((current_input - len_filters[i] + 1) / max_pool_len[i])]))

            self.batch_norm.append(
                nn.BatchNorm1d(N_filters, int((current_input - len_filters[i] + 1) / max_pool_len[i]),
                               momentum=0.05))

            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filters, len_filters))

            else:
                self.conv.append(nn.Conv1d(N_filters[i - 1], N_filters[i], len_filters[i]))

            current_input = int((current_input - len_filters[i] + 1) / max_pool_len[i])

        self.out_dim = current_input * N_filters

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.use_laynorm_inp):
            x = self.layer_norm0((x))

        if bool(self.use_batchnorm_inp):
            x = self.batch_norm0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_lay):

            if self.use_laynorm[i]:
                x = self.dropout[i](
                    self.activations[i](self.layer_norm[i](F.max_pool1d(self.conv[i](x), self.max_pool_len[i]))))

            if self.use_batchnorm[i]:
                x = self.dropout[i](
                    self.activations[i](self.batch_norm[i](F.max_pool1d(self.conv[i](x), self.max_pool_len[i]))))

            if self.use_batchnorm[i] == False and self.use_laynorm[i] == False:
                x = self.dropout[i](self.activations[i](F.max_pool1d(self.conv[i](x), self.max_pool_len[i])))

        x = x.view(batch, -1)

        return x
