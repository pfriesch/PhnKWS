import torch

from torch import nn

from base.base_model import BaseModel

from nn_.net_modules.MLPModule import MLPModule


class MLP(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, outputs,
                 dnn_lay=[1024, 1024, 1024, 1024, 1024],
                 dnn_drop=[0.15, 0.15, 0.15, 0.15, 0.15],
                 dnn_use_laynorm_inp=False,
                 dnn_use_batchnorm_inp=False,
                 dnn_use_batchnorm=[True, True, True, True, True],
                 dnn_use_laynorm=[False, False, False, False, False],
                 dnn_act=['relu', 'relu', 'relu', 'relu', 'relu'],
                 batch_ordering="TNCL"
                 ):
        super(MLP, self).__init__()
        self.input_feat_name = input_feat_name
        self.input_feat_length = input_feat_length
        self.context_left = 5
        self.context_right = 5

        self.MLP = MLPModule(input_feat_length * (self.context_left + self.context_right + 1),
                             dnn_lay=dnn_lay,
                             dnn_drop=dnn_drop,
                             dnn_use_laynorm_inp=dnn_use_laynorm_inp,
                             dnn_use_batchnorm_inp=dnn_use_batchnorm_inp,
                             dnn_use_batchnorm=dnn_use_batchnorm,
                             dnn_use_laynorm=dnn_use_laynorm,
                             dnn_act=dnn_act)

        self.output_layers = nn.ModuleDict({})
        self.out_names = []
        for _output_name, _output_num in outputs.items():
            self.out_names.append(_output_name)

            self.output_layers[_output_name] = MLPModule(self.MLP.out_dim,
                                                         dnn_lay=[_output_num],
                                                         dnn_drop=[0.0],
                                                         dnn_use_laynorm_inp=False,
                                                         dnn_use_batchnorm_inp=False,
                                                         dnn_use_batchnorm=[False],
                                                         dnn_use_laynorm=[False],
                                                         dnn_act=["log_softmax"])
        assert batch_ordering in ['NCL', 'TNCL']
        self.batch_ordering = batch_ordering

    def info(self):
        return f" context: {self.context_left}, {self.context_right}" \
               + f" receptive_field: {self.context_left + self.context_right}"

    def forward(self, _input):
        x = _input[self.input_feat_name]

        if self.batch_ordering == 'TNCL':
            T, batch, feats, context = x.shape
            assert feats == self.input_feat_length
            assert context == self.context_left + self.context_right + 1
            x = x.view(T * batch, feats * context)
        elif self.batch_ordering == 'NCL':
            batch, feats, context = x.shape
            assert feats == self.input_feat_length
            assert context == self.context_left + self.context_right + 1
            x = x.view(batch, feats * context)
        else:
            raise ValueError

        out_dnn = self.MLP(x)

        if self.batch_ordering == 'TNCL':

            out_dict = {}
            for _output_name, _output_layers in self.output_layers.items():
                out_dict[_output_name] = _output_layers(out_dnn).view(T, batch, -1)
        elif self.batch_ordering == 'NCL':
            out_dict = {}
            for _output_name, _output_layers in self.output_layers.items():
                # we do not have T
                out_dict[_output_name] = _output_layers(out_dnn).view(batch, -1, 1)
        else:
            raise RuntimeError

        return out_dict

    # def load_wired_state_dict(self, wired_state_dict):
    #     for i in range(5):
    #         _wired_state_dict = {k: v for k, v in wired_state_dict.items() if k.startswith(f"MLP.layers.{i}.")}
    #         self.layers[i].load_wired_state_dict(_wired_state_dict)
    #
    #     _wired_state_dict = {k: v for k, v in wired_state_dict.items() if
    #                          k.startswith("output_layers.out_cd.layers.0.")}
    #     self.output_layers['out_cd'].load_wired_state_dict(_wired_state_dict)

    def load_warm_start(self, state_dict):
        checkpoint = torch.load(state_dict, map_location='cpu')

        state_dict_orig = checkpoint['state_dict']
        state_dict = {k.replace("tdnn", "MLP").replace("MLP.layers.5.", "output_layers.out_cd.layers.0."): v for k, v in
                      state_dict_orig.items()}

        _out_mono_layer_state_dict = {k: v for k, v in self.state_dict().items() if "out_mono" in k}
        state_dict.update(_out_mono_layer_state_dict)

        _out_cd_layer_state_dict = {k: v for k, v in self.state_dict().items() if "out_cd" in k}

        for k, v in state_dict.items():
            if "output_layers.out_cd.layers.0." in k:
                if len(_out_cd_layer_state_dict[k].shape) > 0:
                    _out_cd_layer_state_dict[k][1:] = state_dict[k]

        state_dict.update(_out_cd_layer_state_dict)

        # self.state_dict()
        # for k in checkpoint:
        #     if 'output_layer' in k:
        #         new_t = self.state_dict()[k]
        #         new_t[1:] = checkpoint[k]
        #         checkpoint[k] = new_t

        self.load_state_dict(state_dict)
