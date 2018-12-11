from distutils.util import strtobool

import torch.nn.functional as F
import torch.nn as nn

from neural_networks.utils import act_fun, LayerNorm


class CNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options['cnn_N_filt'].split(',')))

        self.cnn_len_filt = list(map(int, options['cnn_len_filt'].split(',')))
        self.cnn_max_pool_len = list(map(int, options['cnn_max_pool_len'].split(',')))

        self.cnn_act = options['cnn_act'].split(',')
        self.cnn_drop = list(map(float, options['cnn_drop'].split(',')))

        self.cnn_use_laynorm = list(map(strtobool, options['cnn_use_laynorm'].split(',')))
        self.cnn_use_batchnorm = list(map(strtobool, options['cnn_use_batchnorm'].split(',')))
        self.cnn_use_laynorm_inp = strtobool(options['cnn_use_laynorm_inp'])
        self.cnn_use_batchnorm_inp = strtobool(options['cnn_use_batchnorm_inp'])

        self.N_cnn_lay = len(self.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))

            self.bn.append(
                nn.BatchNorm1d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]),
                               momentum=0.05))

            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        x = x.view(batch, -1)

        return x
