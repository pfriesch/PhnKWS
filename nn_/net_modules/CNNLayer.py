from math import ceil

import torch.nn.functional as F
import torch.nn as nn


class ConvLayerTime(nn.Module):
    def __init__(self,
                 # input_dim,
                 prev_N_filter,
                 N_filter,
                 kernel_size,
                 max_pool_len,
                 # use_laynorm,
                 use_batchnorm,
                 activation_fun,
                 dropout):
        super(ConvLayerTime, self).__init__()
        # assert use_laynorm != use_batchnorm

        # self.use_laynorm = use_laynorm
        self.use_batchnorm = use_batchnorm
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fun = activation_fun

        self.max_pool_len = max_pool_len
        self.conv = nn.Conv1d(prev_N_filter, N_filter, kernel_size)
        self.kernel_size = self.conv.kernel_size[0]
        self.dilation = self.conv.dilation[0]
        self.stride = self.conv.stride[0]
        self.padding = self.conv.padding[0]

        # self.layer_norm = nn.LayerNorm([N_filter, (input_dim - kernel_size + 1) // max_pool_len])
        # self.layer_norm = nn.LayerNorm()
        self.batch_norm = nn.BatchNorm1d(N_filter,
                                         # (input_dim - kernel_size + 1) // max_pool_len,
                                         momentum=0.05)

        # self.input_dim = input_dim

    def forward(self, x):
        batch = x.shape[0]
        feat_dim = x.shape[1]
        seq_len = x.shape[2]

        x = self.conv(x)
        x = F.max_pool1d(x, self.max_pool_len)

        if self.use_batchnorm:
            x = self.batch_norm(x)

        x = self.activation_fun(x)
        x = self.dropout(x)

        return x

    def dilated_kernel_size(self):
        return (self.kernel_size - 1) * self.dilation + 1

    def padding(self):
        return self.output_size() - self.raw_output_size()

    def output_size(self, seq_len_in):
        return ((seq_len_in + 2 * self.padding - self.dilation * (
                self.kernel_size - 1) - 1) // self.stride) + 1

    #
    def raw_output_size(self):
        return ceil((self.input_size - (self.dilated_kernel_size() - 1)) / self.stride)

    def output_size(self):
        # TODO figure out pytorch padding type

        if self.padding == 0:
            return self.raw_output_size()
        else:
            # TODO
            raise NotImplementedError
            # return ceil(self.input_size / self.stride)

    def growth_rate(self):
        growth_rate = self.prev_layer.growth_rate() if self.prev_layer is not None else 1
        return self.stride * growth_rate

    def receptive_field(self):
        if self.prev_layer is not None:
            return self.prev_layer.receptive_field() + (
                    (self.kernel_size - 1) * self.dilation) * self.prev_layer.growth_rate()
        else:
            return (self.kernel_size - 1) * self.dilation + 1
