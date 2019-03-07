import torch

from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from base.base_model import BaseModel


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


class GCNN_cd_mono_Net(BaseModel):

    def __init__(self,
                 input_feat_length,
                 input_feat_name,
                 lookahead_context,
                 outputs,
                 num_feats=[50, 50, 50, 50],
                 dropout=[0.15, 0.15, 0.15, 0.15],
                 kernel_size=3,
                 ):
        super(GCNN_cd_mono_Net, self).__init__()
        self.input_feat_name = input_feat_name
        self.batch_ordering = "NCL"

        assert len(num_feats) == len(dropout)

        self.lookahead_context = lookahead_context

        self.layers = nn.ModuleList()

        for feats, dropout in zip(num_feats, dropout):
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

        self.output_layers = nn.ModuleDict({})
        self.out_names = []
        for _output_name, _output_num in outputs.items():
            self.out_names.append(_output_name)

            self.output_layers[_output_name] = nn.ModuleList([
                Conv1d(n_skip_channels, n_skip_channels,
                       bias=False, w_init_gain='relu'),
                Conv1d(n_skip_channels, _output_num,
                       bias=False, w_init_gain='linear')
            ])

        self.context_left, self.context_right = self.context()

        # self.gcnn = GCNN(num_input_feats=self.num_input_feats,
        #                  num_output_feats=gcnn_out_num,
        #                  N_filters=N_filters,
        #                  kernel_sizes=kernel_sizes,
        #                  dropout=dropout[:-1])
        #
        # self.linear_lab_cd = weight_norm(nn.Linear(gcnn_out_num, lab_cd_num), dim=0)
        # self.linear_lab_mono = weight_norm(nn.Linear(gcnn_out_num, lab_mono_num), dim=0)
        #
        # self.context_left, self.context_right = self.context()

    def forward(self, x):
        x = x[self.input_feat_name]
        x = x.permute(1, 2, 0).unsqueeze(2)

        x = self.gcnn(x)

        out_cd = F.log_softmax(self.linear_lab_cd(x), dim=3)
        out_mono = F.log_softmax(self.linear_lab_mono(x), dim=3)
        return {'out_cd': out_cd, 'out_mono': out_mono}

    def context(self):
        return self.gcnn.context()

    def info(self):
        return f" context: {self.context_left}, {self.context_right}" \
               + f" receptive_field: {self.context_left + self.context_right}"

    def forward(self, _input):
        x = _input[self.input_feat_name]
        # [N , C_in, L]

        output_length = x.shape[2] - self.context_left - self.context_right

        x = self.conv_in(x)
        x = torch.tanh(x)

        skip_connection = None
        for layer in self.layers:
            x, _skip_connection = layer(x)
            if skip_connection is None:
                skip_connection = _skip_connection[:, :, -output_length:]
            else:
                skip_connection = _skip_connection[:, :, -output_length:] + skip_connection
                # TODO try multiply with sqrt(0,5) to keep variance at bay
        x = F.relu(skip_connection)

        out_dict = {}
        for _output_name, _output_layers in self.output_layers.items():
            assert len(_output_layers) == 2

            out_dict[_output_name] = F.relu(_output_layers[0](x))
            out_dict[_output_name] = F.log_softmax(_output_layers[1](out_dict[_output_name]), dim=1)

        return out_dict

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
