from distutils.util import strtobool

import torch
import torch.nn as nn

from neural_networks.utils import LayerNorm, act_fun


class liGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop = list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm = list(map(strtobool, options['ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm = list(map(strtobool, options['ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp = strtobool(options['ligru_use_laynorm_inp'])
        self.ligru_use_batchnorm_inp = strtobool(options['ligru_use_batchnorm_inp'])
        self.ligru_orthinit = strtobool(options['ligru_orthinit'])
        self.ligru_act = options['ligru_act'].split(',')
        self.bidir = strtobool(options['ligru_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])
        self.to_do = options['to_do']

        if self.to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.ligru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.ligru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_ligru_lay = len(self.ligru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

            # Activations
            self.act.append(act_fun(self.ligru_act[i]))

            add_bias = True

            if self.ligru_use_laynorm[i] or self.ligru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))

            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.ligru_lay[i]))

            if self.bidir:
                current_input = 2 * self.ligru_lay[i]
            else:
                current_input = self.ligru_lay[i]

        self.out_dim = self.ligru_lay[i] + self.bidir * self.ligru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.ligru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.ligru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.ligru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.ligru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.ligru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.ligru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = (zt * ht + (1 - zt) * hcand)

                if self.ligru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x
