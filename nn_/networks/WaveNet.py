import itertools
import math
import torch

# import asciichartpy
import torch.nn.functional as F
# from tabulate import tabulate
from torch import nn, Tensor

from base.base_model import BaseModel
from nn_.net_modules.WaveNetModules import Conv1d, WaveNetLayer, Conv2d
from nn_.utils.CNN_utils import LayerStats


class WaveNet(BaseModel):
    def __init__(self,
                 input_feat_length,
                 input_feat_name,
                 outputs,
                 lookahead_context,
                 n_layers=28,
                 n_channels=64,
                 max_dilation=4,
                 n_residual_channels=32,
                 n_skip_channels=64,
                 kernel_size=3,
                 bias=True):
        super(WaveNet, self).__init__()
        self.input_feat_name = input_feat_name
        self.batch_ordering = "NCL"

        self.n_layers = n_layers

        self.lookahead_context = lookahead_context

        self.max_dilation = max_dilation

        self.pool_inf_freq_domain = False
        if self.pool_inf_freq_domain:
            self.conv_in = Conv2d(input_feat_length, n_residual_channels, kernel_size_freq=7, kernel_size_time=5)
        else:
            self.conv_in = Conv1d(input_feat_length, n_residual_channels, bias=bias)

        self.layers = nn.ModuleList()
        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):
            dilation = 2 ** (i % loop_factor)

            self.layers.append(
                WaveNetLayer(kernel_size, n_channels, n_residual_channels, n_skip_channels, dilation, bias,
                             no_output_layer=i == n_layers - 1))

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

        self.context_left, self.context_right = self.context

    def info(self):
        return f" context: {self.context_left}, {self.context_right}" \
               + f" receptive_field: {self.context_left + self.context_right}"

    def forward(self, _input):
        if isinstance(_input, dict):
            x = _input[self.input_feat_name]
        elif isinstance(_input, Tensor):
            x = _input
        else:
            raise NotImplementedError
        # [N , C_in, L]

        output_length = x.shape[2] - self.context_left - self.context_right

        if self.pool_inf_freq_domain:
            x = x.unsqueeze(1)
            x = self.conv_in(x)

        else:
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

    @property
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

    # def _cnn_stats(self, input_size):
    #     layers = self._get_layer_stats()
    #
    #     _print_tabular = []
    #     _receptive_field = []
    #     for i in range(len(layers)):
    #         layer = layers[i]
    #
    #         if i > 0:
    #             prev_layer = layers[i - 1]
    #             layer.prev_layer = prev_layer
    #             layer.input_size = prev_layer.output_size()
    #         else:
    #             layer.prev_layer = None
    #             layer.input_size = input_size
    #
    #         _print_tabular.append([i, type(layer).__name__, layer.kernel_size, layer.stride, layer.dilation,
    #                                layer.padding, layer.input_size,
    #                                layer.output_size(), layer.receptive_field()])
    #         _receptive_field.append(layer.receptive_field())
    #     print(tabulate(_print_tabular, headers=(
    #         "Layer #", "Name", "Kernel Size", "Stride", "Dilation", "Padding", "Input Size", "Output Size",
    #         "Receptive Field")))
    #
    #     print(asciichartpy.plot(
    #         list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in _receptive_field))
    #         , cfg={"height": 10}))
    #
    #     return layer.output_size()

    # def load_warm_start(self, path):
    #     # CE mono+cd+phnframe -> CTC phn
    #     state_dict = torch.load(path, map_location='cpu')['state_dict']
    #
    #     state_dict = {k.replace("out_phnframe", "out_phn"): v for k, v in state_dict.items() if
    #                   'out_mono' not in k and 'out_cd' not in k}
    #
    #     self.load_state_dict(state_dict)
    #     print("Loaded state dict for warm start")

    def get_custom_ops_for_counting(self):

        def count_conv1d(m, x, y):
            x = x[0]

            cin = m.in_channels
            cout = m.out_channels
            kh, kw = m.kernel_size
            batch_size = x.size()[0]

            out_h = y.size(2)
            out_w = y.size(3)

            # ops per output element
            # kernel_mul = kh * kw * cin
            # kernel_add = kh * kw * cin - 1
            kernel_ops = kh * kw * cin // m.groups
            bias_ops = 1 if m.bias is not None else 0
            ops_per_element = kernel_ops + bias_ops

            # total ops
            # num_out_elements = y.numel()
            output_elements = batch_size * out_w * out_h * cout
            total_ops = output_elements * ops_per_element

            m.total_ops = torch.Tensor([int(total_ops)])

        return {Conv1d: count_conv1d}

    # def load_warm_start(self, path):
    #     # CE mono+cd+phnframe -> CE phnframe
    #     state_dict = torch.load(path, map_location='cpu')['state_dict']
    #
    #     state_dict = {k: v for k, v in state_dict.items() if
    #                   'out_mono' not in k and 'out_cd' not in k}
    #
    #     self.load_state_dict(state_dict)
    #     print("Loaded state dict for warm start")

    # def load_warm_start(self, path):
    #     # CE cd -> cd+phnframe
    #     state_dict = torch.load(path, map_location='cpu')['state_dict']
    #
    #     state_dict = {k: v for k, v in state_dict.items() if 'out_mono' not in k}
    #     # state_dict.update({k: v for k, v in self.state_dict().items() if 'out_phnframe' in k})
    #
    #     self.load_state_dict(state_dict)
    #     print("Loaded state dict for warm start")

    def load_warm_start(self, path):
        # CE mono+cd -> mono+cd+phnframe
        state_dict = torch.load(path, map_location='cpu')['state_dict']

        state_dict = {k.replace("out_phnframe", "out_phn"): v for k, v in state_dict.items() if
                      'out_mono' not in k and 'out_cd' not in k}

        self.load_state_dict(state_dict)
        print("Loaded state dict for warm start")

    # def load_warm_start(self, path):
    # #same arch
    #     state_dict = torch.load(path)['state_dict']
    #     # state_dict = {k: v for k, v in state_dict.items() if not 'out_mono' in k}
    #
    #     self.load_state_dict(state_dict)
    #     print("Loaded state dict for warm start")
    #
    # def load_warm_start(self, path):
    #     # CE -> CTC
    #
    #     state_dict = torch.load(path, map_location='cpu')['state_dict']
    #     _state_dict_old = {k: v for k, v in state_dict.items() if not 'output_layers' in k}
    #
    #     _state_dict_init = {k: v for k, v in self.state_dict().items() if 'output_layers' in k}
    #     _state_dict_old.update(_state_dict_init)
    #
    #     self.load_state_dict(_state_dict_old)
    #     print("Loaded state dict for warm start")
    #     # for param in self.layers.parameters():
    #     #     param.requires_grad = False
    #     # for param in self.conv_in.parameters():
    #     #     param.requires_grad = False
    #
    #     # print("Froze Layers")

    # def load_warm_start(self, path):
    #     # CE -> CTC
    #
    #     state_dict = torch.load(path)['state_dict']
    #     _state_dict_old = {k: v for k, v in state_dict.items() if not 'output_layers' in k}
    #
    #     _state_dict_init = {k: v for k, v in self.state_dict().items() if 'output_layers' in k}
    #     _state_dict_old.update(_state_dict_init)
    #
    #     self.load_state_dict(_state_dict_old)
    #     print("Loaded state dict for warm start")

# if __name__ == '__main__':
#     _x = torch.zeros(5, 40, 16).random_()
#     _x = {'fbank': _x}
#     wn = WaveNet(input_feat_length=40, input_feat_name='fbank', lab_mono_num=4, lab_cd_num=19, lookahead_context=0,
#                  n_layers=4, max_dilation=20, n_residual_channels=16, n_skip_channels=32, kernel_size=2)
#     _out = wn(_x)
#
#     print(_out)
