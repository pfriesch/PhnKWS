import torch.nn as nn

from nn_.net_modules.ConvLayer import ConvLayer
from nn_.net_modules.utils import act_fun
from nn_.utils.CNN_utils import cnn_receptive_field


class CNN(nn.Module):

    def __init__(self, N_in_feats,
                 N_filters,
                 kernel_sizes,
                 max_pool_len,
                 activation,
                 dropout):
        super(CNN, self).__init__()
        self.N_in_feats = N_in_feats

        self.N_lay = len(N_filters)

        self.layers = nn.ModuleList([])

        self.receptive_field = 32

        prev_N_filters = N_in_feats

        for i in range(self.N_lay):
            if i != 0:
                prev_N_filters = N_filters[i - 1]

            self.layers.append(ConvLayer(
                prev_N_filters, N_filters[i], kernel_sizes[i],
                max_pool_len[i],
                act_fun(activation[i]),
                dropout[i]
            ))

        assert self.receptive_field == cnn_receptive_field(self), \
            "receptive_field mismatch, set the actual receptive field of {}".format(cnn_receptive_field(self))

        self.N_out_feats = self.layers[-1].N_out_filters

    def forward(self, x):

        T = x.shape[0]
        batch = x.shape[1]
        assert x.shape[2] == self.N_in_feats
        assert x.shape[3] == self.receptive_field

        x = x.view(T * batch, self.N_in_feats, self.receptive_field)

        for i in range(self.N_lay):
            x = self.layers[i](x)

        L_out = 1
        assert batch * T == x.shape[0]
        assert x.shape[1] == self.N_out_feats
        assert x.shape[2] == L_out

        x = x.view(T, batch, self.N_out_feats, L_out)

        return x
