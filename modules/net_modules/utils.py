##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################


import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    elif act_type == "tanh":
        return nn.Tanh()

    elif act_type == "sigmoid":
        return nn.Sigmoid()

    elif act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    elif act_type == "elu":
        return nn.ELU()

    elif act_type == "softmax":
        return nn.LogSoftmax(dim=1)  # TODO fix this ambiguity

    elif act_type == "log_softmax":
        return nn.LogSoftmax(dim=1)

    elif act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!
    else:
        raise ValueError("Activation function {} not found!".format(act_type))


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
