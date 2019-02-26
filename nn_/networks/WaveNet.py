import itertools
import math
import torch

import asciichartpy
import torch.nn.functional as F
from tabulate import tabulate
from torch import nn

from base.base_model import BaseModel
from nn_.net_modules.WaveNetModules import Conv1d, WaveNetLayer
from nn_.utils.CNN_utils import LayerStats


class WaveNet(BaseModel):
    def __init__(self, input_feat_length, input_feat_name,
                 lab_cd_num,
                 lab_mono_num,
                 lookahead_context=5,
                 n_layers=24,
                 max_dilation=4,
                 n_residual_channels=16,
                 n_skip_channels=32,
                 kernel_size=3):
        super(WaveNet, self).__init__()
        self.input_feat_name = input_feat_name
        self.n_layers = n_layers
        # assert n_layers == 24
        self.lookahead_context = lookahead_context

        self.max_dilation = max_dilation
        # assert max_dilation == 8

        self.layers = nn.ModuleList()

        self.conv_in = Conv1d(input_feat_length, n_residual_channels)

        self.conv_out_cd = Conv1d(n_skip_channels, lab_cd_num,
                                  bias=False, w_init_gain='relu')
        self.conv_end_cd = Conv1d(lab_cd_num, lab_cd_num,
                                  bias=False, w_init_gain='linear')
        self.conv_out_mono = Conv1d(n_skip_channels, lab_mono_num,
                                    bias=False, w_init_gain='relu')
        self.conv_end_mono = Conv1d(lab_mono_num, lab_mono_num,
                                    bias=False, w_init_gain='linear')

        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):
            dilation = 2 ** (i % loop_factor)

            self.layers.append(
                WaveNetLayer(kernel_size, n_residual_channels, n_skip_channels, dilation,
                             no_output_layer=i == n_layers - 1))

        self.context_left, self.context_right = self.context()

        self.batch_ordering = "BCL"

    def forward(self, _input):
        x = _input[self.input_feat_name]
        # [N , C_in, L]

        # output_length = x.shape[2] - self.context_left - self.context_right

        x = self.conv_in(x)
        x = torch.tanh(x)

        skip_connection = None
        for layer in self.layers:
            x, _skip_connection = layer(x)
            if skip_connection is None:
                skip_connection = _skip_connection[:, :, -1:]
            else:
                skip_connection = _skip_connection[:, :, -1:] + skip_connection
                # TODO check for variance and and multiply wiht sqrt(0,5)
        x = skip_connection

        x_cd = F.relu(x)
        x_cd = self.conv_out_cd(x_cd)
        x_cd = F.relu(x_cd)
        x_cd = self.conv_end_cd(x_cd)
        x_cd = F.log_softmax(x_cd, dim=1)
        assert x_cd.shape[2] == 1
        x_cd = x_cd.squeeze(2)

        x_mono = F.relu(x)
        x_mono = self.conv_out_mono(x_mono)
        x_mono = F.relu(x_mono)
        x_mono = self.conv_end_mono(x_mono)
        x_mono = F.log_softmax(x_mono, dim=1)
        assert x_mono.shape[2] == 1
        x_mono = x_mono.squeeze(2)

        return {'out_cd': x_cd, 'out_mono': x_mono}

    def context(self):
        _receptive_field = self._receptive_field()
        _receptive_field_total = _receptive_field[-1]
        context_right = self.lookahead_context
        context_left = _receptive_field_total - self.lookahead_context - 1
        assert context_left + context_right + 1 == _receptive_field_total
        return context_left, context_right

    def _get_layer_stats(self):
        layers = []
        for _layer in self.layers:
            layers.append(
                LayerStats(_layer.in_layer.conv.kernel_size[0],
                           _layer.in_layer.conv.stride[0],
                           _layer.in_layer.conv.dilation[0],
                           _layer.in_layer.conv.padding[0],
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

# if __name__ == '__main__':
#     _x = torch.zeros(5, 40, 16).random_()
#     _x = {'fbank': _x}
#     wn = WaveNet(input_feat_length=40, input_feat_name='fbank', lab_mono_num=4, lab_cd_num=19, lookahead_context=0,
#                  n_layers=4, max_dilation=20, n_residual_channels=16, n_skip_channels=32, kernel_size=2)
#     _out = wn(_x)
#
#     print(_out)
