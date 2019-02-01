import itertools

import torch
from torch import nn
from torch.nn.utils import weight_norm

from modules.utils.CNN_utils import LayerStats


def gated_linear_unit(x, dim=1):
    # default is dim 1 which is C for conv2d
    dim_len = x.shape[dim]
    x, x_gate = torch.split(x, dim_len // 2, dim)

    return x * torch.sigmoid(x_gate)


class GatedConv2D(nn.Module):

    def __init__(self, n_in_filters, n_out_filters, kernel_size, p_dropout=0.0):
        super(GatedConv2D, self).__init__()
        self.conv = weight_norm(nn.Conv2d(n_in_filters, n_out_filters, kernel_size), dim=0)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.conv(x)
        x = gated_linear_unit(x)
        x = self.dropout(x)
        return x


class GCNN(nn.Module):

    def __init__(self, num_input_feats,
                 num_output_feats,
                 N_filters,
                 kernel_sizes,
                 dropout):
        super(GCNN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(N_filters)):
            if i == 0:
                prev_N_filters = num_input_feats
            else:
                prev_N_filters = N_filters[i - 1] // 2
                assert 2 * prev_N_filters == N_filters[i - 1], "N_filters not divisible by 2"

            self.layers.append(GatedConv2D(prev_N_filters, N_filters[i], kernel_sizes[i], dropout[i]))

        prev_N_filters = N_filters[-1] // 2
        assert 2 * prev_N_filters == N_filters[-1], "N_filters not divisible by 2"

        self.linear = weight_norm(nn.Linear(prev_N_filters, num_output_feats * 2), dim=0)
        self.dropout = nn.Dropout(p=dropout[-1])

        self.out_dim = num_output_feats

    def forward(self, x):

        for _i, layer in enumerate(self.layers):
            x = layer(x)

        x = x.permute(2, 0, 3, 1)
        # reorder
        x = gated_linear_unit(self.linear(x), dim=3)
        x = self.dropout(x)

        return x

    def context(self):
        _receptive_field = self._receptive_field()
        _receptive_field_total = _receptive_field[-1]
        context_left = _receptive_field_total // 2  # TODO less lookahead
        context_right = _receptive_field_total // 2
        assert context_left + context_right + 1 == _receptive_field_total
        return context_left, context_right

    def _get_layer_stats(self):
        layers = []
        for _layer in self.layers:
            layers.append(
                LayerStats(_layer.conv.kernel_size[1],
                           _layer.conv.stride[1],
                           _layer.conv.dilation[1],
                           _layer.conv.padding[1],
                           name=type(_layer).__name__))
        return layers

    def _receptive_field(self):
        layers = self._get_layer_stats()

        _receptive_field = []
        for i in range(len(layers)):
            layer = layers[i]

            if i > 0:
                prev_layer = layers[i - 1]
                layer.prev_layer = prev_layer
            else:
                layer.prev_layer = None

            _receptive_field.append(layer.receptive_field())
        return _receptive_field

    def _cnn_stats(self, input_size):
        layers = self._get_layer_stats()

        _print_tabular = []
        _receptive_field = []
        for i in range(len(layers)):
            layer = layers[i]

            if i > 0:
                prev_layer = layers[i - 1]
                layer.prev_layer = prev_layer
                layer.input_size = prev_layer.output_size()
            else:
                layer.prev_layer = None
                layer.input_size = input_size

            _print_tabular.append([i, type(layer).__name__, layer.kernel_size, layer.stride, layer.dilation,
                                   layer.padding, layer.input_size,
                                   layer.output_size(), layer.receptive_field()])
            _receptive_field.append(layer.receptive_field())
        print(tabulate(_print_tabular, headers=(
            "Layer #", "Name", "Kernel Size", "Stride", "Dilation", "Padding", "Input Size", "Output Size",
            "Receptive Field")))

        print(asciichartpy.plot(
            list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in _receptive_field))
            , cfg={"height": 10}))

        return layer.output_size()
