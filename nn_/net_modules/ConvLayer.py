import torch.nn as nn
import torch.nn.functional as F

from nn_.utils.CNN_utils import LayerStats
from base.base_layer import BaseLayer


class ConvLayer(BaseLayer):
    def __init__(self,
                 N_in_filters,
                 N_filters,
                 kernel_size,
                 max_pool_len,
                 activation_fun,
                 dropout):
        super(ConvLayer, self).__init__()

        self.N_in_filters = N_in_filters

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
        if self.max_pool_len > 0:
            x = F.max_pool1d(x, self.max_pool_len)

        x = self.activation_fun(x)
        x = self.dropout(x)

        return x

    def get_layer_stats(self):
        _layers = [
            LayerStats(self.conv.kernel_size[0], self.conv.stride[0], self.conv.dilation[0], self.conv.padding[0],
                       name=type(self.conv).__name__)]
        if self.max_pool_len > 0:
            _layers += [LayerStats(self.max_pool_len, self.max_pool_len, 1, 0,
                                   name=str(type(self.conv).__name__) + "_max_pool")]

        return _layers
