import torch.nn as nn
import torch.nn.functional as F

from _nn.utils.CNN_utils import LayerStats
from base.base_layer import BaseLayer


class ConvLayer(BaseLayer):
    def __init__(self,
                 # input_dim,
                 N_in_filters,
                 N_filters,
                 kernel_size,
                 max_pool_len,
                 use_laynorm,
                 use_batchnorm,
                 activation_fun,
                 dropout):
        super(ConvLayer, self).__init__()
        assert use_laynorm != use_batchnorm

        self.N_in_filters = N_in_filters

        self.use_laynorm = use_laynorm

        self.dropout = nn.Dropout(p=dropout)
        self.activation_fun = activation_fun

        self.max_pool_len = max_pool_len
        self.conv = nn.Conv1d(N_in_filters, N_filters, kernel_size)

        self.N_out_filters = N_filters


    def forward(self, x):
        # [B, C, T]
        batch = x.shape[0]
        assert x.shape[1] == self.N_in_filters
        seq_len = x.shape[2]

        x = self.conv(x)
        x = F.max_pool1d(x, self.max_pool_len)

        if self.use_laynorm:
            x = self.layer_norm(x)

        x = self.activation_fun(x)
        x = self.dropout(x)

        return x

    def get_layer_stats(self):
        return [
            LayerStats(self.conv.kernel_size[0], self.conv.stride[0], self.conv.dilation[0], self.conv.padding[0],
                       name=type(self.conv).__name__),
            LayerStats(self.max_pool_len, self.max_pool_len, 1, 0,
                       name=str(type(self.conv).__name__) + "_max_pool")]
