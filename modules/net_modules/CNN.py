import torch.nn.functional as F
import torch.nn as nn

from modules.net_modules.utils import LayerNorm, act_fun


class Conv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.l_out = lambda l_in: ((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)


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

        self.max_pool_len = max_pool_len
        self.conv = Conv1d(prev_N_filter, N_filter, kernel_size)

        self.layer_norm = nn.LayerNorm([N_filter, (input_dim - kernel_size + 1) // max_pool_len])
        # self.layer_norm = nn.LayerNorm()
        self.batch_norm = nn.BatchNorm1d(N_filter,
                                         (input_dim - kernel_size + 1) // max_pool_len,
                                         momentum=0.05)

        self.input_dim = input_dim

    def forward(self, x):
        batch = x.shape[0]
        x_dim = x.shape[1]
        seq_len = x.shape[2]
        assert x_dim == self.input_dim

        # if bool(self.use_laynorm_inp):
        #     x = self.layer_norm0((x))
        #
        # if bool(self.use_batchnorm_inp):
        #     x = self.batch_norm0((x))

        # x = x.permute(0, 2, 1)
        # x = x.view(batch, feat_dim, seq_len)

        # for i in range(self.N_lay):

        x = self.conv(x)
        x = F.max_pool1d(x, self.max_pool_len)

        if self.use_laynorm:
            x = self.layer_norm(x)
        elif self.use_batchnorm:
            x = self.batch_norm(x)

        x = self.activation_fun(x)
        x = self.dropout(x)

        # x = x.view(batch, -1)

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
        # self.use_batchnorm = use_batchnorm
        # self.use_laynorm = use_laynorm
        self.use_laynorm_inp = use_laynorm_inp
        self.use_batchnorm_inp = use_batchnorm_inp

        self.N_lay = len(N_filters)
        # self.conv = nn.ModuleList([])
        # self.batch_norm = nn.ModuleList([])
        # self.layer_norm = nn.ModuleList([])
        # self.activations = nn.ModuleList([])
        # self.dropout = nn.ModuleList([])

        self.layers = nn.ModuleList([])

        if self.use_laynorm_inp:
            self.layer_norm0 = LayerNorm(self.num_input_feats * self.context)

        if self.use_batchnorm_inp:
            raise NotImplementedError
            # self.batch_norm0 = nn.BatchNorm1d([num_input], momentum=0.05)

        current_input = self.num_input_feats

        for i in range(self.N_lay):
            if i == 0:
                prev_N_filters = num_input_feats
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

        seq_len = x.shape[0]
        batch = x.shape[1]
        context_dim = x.shape[2]
        feat_dim = x.shape[3]

        x = x.view(seq_len, batch, -1)
        x = x.permute(1, 0, 2)

        batch = x.shape[0]
        seq_len = x.shape[1]
        feat_dim = x.shape[2]
        assert feat_dim == self.num_input_feats * (self.context + 1)

        if bool(self.use_laynorm_inp):
            x = self.layer_norm0((x))

        if bool(self.use_batchnorm_inp):
            x = self.batch_norm0((x))

        x = x.permute(0, 2, 1)
        # x = x.contiguous().view(batch, feat_dim, seq_len)

        for i in range(self.N_lay):
            x = self.layers[i](x)

        x = x.view(batch, -1)

        return x
