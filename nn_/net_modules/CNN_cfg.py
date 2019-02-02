import torch.nn.functional as F
import torch.nn as nn

from nn_.net_modules.utils import LayerNorm, act_fun


class CNNLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 prev_N_filter,
                 N_filter,
                 kernel_size,
                 max_pool_len,
                 use_laynorm,
                 use_batchnorm,
                 activation_fun,
                 dropout):
        super(CNNLayer, self).__init__()
        assert use_laynorm != use_batchnorm

        self.use_laynorm = use_laynorm
        self.use_batchnorm = use_batchnorm
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fun = activation_fun

        self.prev_N_filter = prev_N_filter

        self.max_pool_len = max_pool_len
        self.conv = nn.Conv1d(prev_N_filter, N_filter, kernel_size)

        self.layer_norm = LayerNorm([N_filter, (input_dim - kernel_size + 1) // max_pool_len])

        self.input_dim = input_dim

    def forward(self, x):
        batch = x.shape[0]
        assert x.shape[1] == self.prev_N_filter
        x_dim = x.shape[2]
        assert x_dim == self.input_dim

        x = self.conv(x)
        x = F.max_pool1d(x, self.max_pool_len)

        if self.use_laynorm:
            x = self.layer_norm(x)

        x = self.activation_fun(x)
        x = self.dropout(x)

        return x


class CNN(nn.Module):

    def __init__(self, num_input_feats,
                 input_context,
                 N_filters,
                 kernel_sizes,
                 max_pool_len,
                 use_laynorm,
                 use_batchnorm,
                 use_laynorm_inp,
                 use_batchnorm_inp,
                 activation,
                 dropout):
        super(CNN, self).__init__()
        self.num_input_feats = num_input_feats
        self.context = input_context
        self.use_laynorm_inp = use_laynorm_inp
        self.use_batchnorm_inp = use_batchnorm_inp

        self.N_lay = len(N_filters)

        self.layers = nn.ModuleList([])

        if self.use_laynorm_inp:
            self.layer_norm0 = LayerNorm(self.num_input_feats * self.context)

        if self.use_batchnorm_inp:
            raise NotImplementedError
            # self.batch_norm0 = nn.BatchNorm1d([num_input], momentum=0.05)

        current_input = self.num_input_feats

        for i in range(self.N_lay):
            if i == 0:
                prev_N_filters = 1
            else:
                prev_N_filters = N_filters[i - 1]

            self.layers.append(CNNLayer(
                current_input, prev_N_filters, N_filters[i], kernel_sizes[i],
                max_pool_len[i], use_laynorm[i], use_batchnorm[i],
                act_fun(activation[i]),
                dropout[i]
            ))

            current_input = (current_input - kernel_sizes[i] + 1) // max_pool_len[i]

        self.out_dim = current_input * N_filters[-1]

    def forward(self, x):

        batch = x.shape[0]
        assert x.shape[1] == 1
        n_feats = x.shape[2]  # feats concatenated
        assert n_feats == self.num_input_feats

        if bool(self.use_laynorm_inp):
            x = self.layer_norm0((x))

        if bool(self.use_batchnorm_inp):
            x = self.batch_norm0((x))

        for i in range(self.N_lay):
            x = self.layers[i](x)

        x = x.view(batch, -1)

        return x
