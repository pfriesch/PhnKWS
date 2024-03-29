import torch
import torch.nn as nn
import numpy as np

from nn_.net_modules.utils import act_fun, LayerNorm


class MLPLayer(nn.Module):

    def __init__(self, input_dim, dnn_lay, dnn_drop, dnn_act, dnn_use_laynorm, dnn_use_batchnorm):
        super(MLPLayer, self).__init__()
        self.input_dim = input_dim
        self.dnn_use_laynorm = dnn_use_laynorm
        self.dnn_use_batchnorm = dnn_use_batchnorm
        self.drop = nn.Dropout(p=dnn_drop)
        self.act = act_fun(dnn_act)

        add_bias = True

        # layer norm initialization
        self.ln = LayerNorm(dnn_lay)
        self.bn = nn.BatchNorm1d(dnn_lay, momentum=0.05)

        if self.dnn_use_laynorm or self.dnn_use_batchnorm:
            add_bias = False

        # Linear operations
        self.wx = nn.Linear(input_dim, dnn_lay, bias=add_bias)

        # weight initialization
        self.wx.weight = torch.nn.Parameter(torch.Tensor(dnn_lay, input_dim).uniform_(
            -np.sqrt(0.01 / (input_dim + dnn_lay)), np.sqrt(0.01 / (input_dim + dnn_lay))))
        self.wx.bias = torch.nn.Parameter(torch.zeros(dnn_lay))

    def forward(self, x):
        batch_size = x.shape[0]
        feat = x.shape[1]
        assert self.input_dim == feat

        x = self.wx(x)

        if self.dnn_use_laynorm:
            raise NotImplementedError("given that we use a sequence now this need to change")
            # x = self.ln(x)

        if self.dnn_use_batchnorm:
            x = self.bn(x)

        x = self.act(x)
        x = self.drop(x)
        return x


class MLPModule(nn.Module):
    def __init__(self, input_dim,
                 dnn_lay,
                 dnn_drop,
                 dnn_use_batchnorm,
                 dnn_use_laynorm,
                 dnn_use_laynorm_inp,
                 dnn_use_batchnorm_inp,
                 dnn_act):
        super(MLPModule, self).__init__()

        assert len(dnn_lay) == len(dnn_drop) == len(dnn_use_batchnorm) == len(dnn_use_laynorm) == len(dnn_act), \
            (len(dnn_lay), len(dnn_drop), len(dnn_use_batchnorm), len(dnn_use_laynorm), len(dnn_act))
        assert isinstance(input_dim, int)
        assert [isinstance(elem, int) for elem in dnn_lay]
        assert [isinstance(elem, float) for elem in dnn_drop]
        assert [isinstance(elem, bool) for elem in dnn_use_batchnorm]
        assert [isinstance(elem, bool) for elem in dnn_use_laynorm]
        assert isinstance(dnn_use_laynorm_inp, bool)
        assert isinstance(dnn_use_batchnorm_inp, bool)
        assert [isinstance(elem, str) for elem in dnn_act]

        self.input_dim = input_dim
        self.dnn_lay = dnn_lay
        self.dnn_drop = dnn_drop
        self.dnn_use_batchnorm = dnn_use_batchnorm
        self.dnn_use_laynorm = dnn_use_laynorm
        self.dnn_use_laynorm_inp = dnn_use_laynorm_inp
        self.dnn_use_batchnorm_inp = dnn_use_batchnorm_inp
        self.dnn_act = dnn_act

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.layers = nn.ModuleList([])

        self.N_dnn_lay = len(dnn_lay)

        current_input = input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):
            self.layers.append(
                MLPLayer(current_input,
                         dnn_lay[i],
                         dnn_drop[i],
                         dnn_act[i],
                         dnn_use_laynorm[i],
                         dnn_use_batchnorm[i]))
            current_input = dnn_lay[i]

        self.out_dim = current_input

    def forward(self, x):

        # Applying Layer/Batch Norm
        if self.dnn_use_laynorm_inp:
            raise NotImplementedError("given that we use a sequence now this need to change")
            # x = self.ln0((x))

        if self.dnn_use_batchnorm_inp:
            raise NotImplementedError("given that we use a sequence now this need to change")
            # x = self.bn0((x))

        for i in range(self.N_dnn_lay):
            x = self.layers[i](x)

        return x

    def load_cfg(self):

        final_architecture2 = torch.load(
            "/mnt/data/pytorch-kaldi_cfg/exp/libri_LSTM_fbank/exp_files/final_architecture2.pkl", map_location='cpu')[
            'model_par']

        _state_Stuff = {k.replace("layers.0.", ""): k for k in self.state_dict()}
        _final_architecture2 = {_state_Stuff[k.replace(".0.", ".")]: v for k, v in final_architecture2.items()}

        self.load_state_dict(_final_architecture2)

# if __name__ == '__main__':
