import torch.nn as nn

from _nn.net_modules.ConvLayer import ConvLayer
from _nn.net_modules.utils import act_fun
from _nn.utils.CNN_utils import cnn_context


class CNN(nn.Module):

    def __init__(self, N_in_feats,
                 # N_out_feats,
                 N_filters,
                 kernel_sizes,
                 max_pool_len,
                 use_laynorm,
                 use_batchnorm,
                 activation,
                 dropout):
        super(CNN, self).__init__()
        self.N_in_feats = N_in_feats
        # self.N_out_feats = N_out_feats

        self.N_lay = len(N_filters)

        self.layers = nn.ModuleList([])

        for i in range(self.N_lay):
            if i == 0:
                prev_N_filters = N_in_feats
            else:
                prev_N_filters = N_filters[i - 1]

            self.layers.append(ConvLayer(
                prev_N_filters, N_filters[i], kernel_sizes[i],
                max_pool_len[i], use_laynorm[i], use_batchnorm[i],
                act_fun(activation[i]),
                dropout[i]
            ))

            # current_input = (current_input - kernel_sizes[i] + 1) // max_pool_len[i]

        self.context = cnn_context(self)

        self.N_out_feats = self.layers[-1].N_out_filters

    def forward(self, x):

        batch = x.shape[0]
        assert x.shape[1] == self.N_in_feats
        assert x.shape[2] == self.context
        # [ B, C_in, L_context]

        for i in range(self.N_lay):
            x = self.layers[i](x)

        l_out = 1
        assert batch == x.shape[0]
        assert x.shape[1] == self.N_out_feats
        assert x.shape[2] == l_out
        # [ B, C_out, L_out]

        return x
