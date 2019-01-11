import torch
import torch.nn as nn
import numpy as np

from nets.modules.utils import act_fun, LayerNorm


class MLP(nn.Module):
    def __init__(self, inp_dim, dnn_lay, dnn_drop, dnn_use_batchnorm, dnn_use_laynorm, dnn_use_laynorm_inp,
                 dnn_use_batchnorm_inp, dnn_act):
        super(MLP, self).__init__()

        self.input_dim = inp_dim
        self.dnn_lay = dnn_lay
        self.dnn_drop = dnn_drop
        self.dnn_use_batchnorm = dnn_use_batchnorm
        self.dnn_use_laynorm = dnn_use_laynorm
        self.dnn_use_laynorm_inp = dnn_use_laynorm_inp
        self.dnn_use_batchnorm_inp = dnn_use_batchnorm_inp
        self.dnn_act = dnn_act

        assert [isinstance(elem, int) for elem in self.dnn_lay]
        assert [isinstance(elem, float) for elem in self.dnn_drop]
        assert [isinstance(elem, bool) for elem in self.dnn_use_batchnorm]
        assert [isinstance(elem, bool) for elem in self.dnn_use_laynorm]
        assert isinstance(self.dnn_use_laynorm_inp, bool)
        assert isinstance(self.dnn_use_batchnorm_inp, bool)
        assert [isinstance(elem, str) for elem in self.dnn_act]

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                -np.sqrt(0.01 / (current_input + self.dnn_lay[i])), np.sqrt(0.01 / (current_input + self.dnn_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.dnn_use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))

        return x
