from torch import nn
import torch


class Conv2d(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """

    def __init__(self, in_channels, out_channels, kernel_size_freq=1, kernel_size_time=1, stride_freq=1, stride_time=1,
                 dilation_freq=1, dilation_time=1, bias=False, w_init_gain='linear'):
        super(Conv2d, self).__init__()
        self.kernel_size = (kernel_size_freq, kernel_size_time)
        self.dilation = (dilation_freq, dilation_time)
        self.stride = (stride_freq, stride_time)

        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=self.kernel_size, stride=self.stride,
                                    dilation=self.dilation, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class Conv1d(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv1d, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        # if self.is_causal:
        #     padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
        #     signal = torch.nn.functional.pad(signal, padding)
        return self.conv(signal)


class WaveNetLayer(nn.Module):
    """
                    ^ Output
                    |
                    |
    +-----------------------------------+
    |               |                   |
    |             +---+                 |
    |   +-------->| + |                 |
    |   |         +---+                 |
    |   |           |                   |
    |   |         +-+-+                 |   SkipConnection
    |   |         |1x1|------------------------->
    |   |         +-+-+                 |
    |   |           ^                   |
    |   |          +-+                  |
    |   |   +----->|X|<---------+       |
    |   | +----+   +-+      +-------+   |
    |   | |tanh|            |sigmoid|   |
    |   | +-+--+            +--+----+   |
    |   |   ^  +-----------+   |        |
    |   |   +--+DilatedCon<----+        |
    |   |      +----+------+            |
    |   |           ^                   |
    |   +-----------+                   |
    +-----------------------------------+
                    |
                    |
                    + Input

    """

    def __init__(self, kernel_size, n_channels, n_residual_channels, n_skip_channels, dilation, no_output_layer=False):
        super().__init__()
        self.n_channels = n_channels
        assert n_channels % 2 == 0
        self.in_layer = Conv1d(n_residual_channels, n_channels,
                               kernel_size=kernel_size, dilation=dilation,
                               w_init_gain='tanh', is_causal=True)

        self.no_output_layer = no_output_layer
        if not self.no_output_layer:
            self.res_layer = Conv1d(n_channels // 2, n_residual_channels,
                                    w_init_gain='linear')
        self.skip_layer = Conv1d(n_channels // 2, n_skip_channels,
                                 w_init_gain='relu')

    def forward(self, x):
        # TODO r9y9 adds dropout to x
        in_act = self.in_layer(x)
        t_act = torch.tanh(in_act[:, :self.n_channels // 2, :])
        s_act = torch.sigmoid(in_act[:, self.n_channels // 2:, :])
        acts = t_act * s_act

        if not self.no_output_layer:
            res_acts = self.res_layer(acts)
            x = res_acts + x[:, :, -res_acts.shape[2]:]
            # TODO r9y9 adds here for variance?: * math.sqrt(0.5)

        s = self.skip_layer(acts)

        if not self.no_output_layer:

            return x, s
        else:
            return None, s
