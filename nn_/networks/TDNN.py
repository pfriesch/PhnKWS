import torch

from torch import nn

from base.base_model import BaseModel

from nn_.net_modules.utils import act_fun


class TDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, activation, dropout=0.0, use_laynorm=False, use_batchnorm=False,
                 kernel_size=1,
                 stride=1):
        super(TDNNLayer, self).__init__()
        self.in_channels = in_channels
        self.drop = nn.Dropout(p=dropout)
        self.act = act_fun(activation)

        assert not (use_laynorm and use_batchnorm)
        if use_laynorm:
            # pytorch-kaldi uses LayerNorm here, but their batches have [N * L,C] and we have [N,C,L] so InstanceNorm is the eqivalent here
            self.ln = nn.InstanceNorm1d(out_channels, momentum=0.0, affine=True)
            add_bias = False
        elif use_batchnorm:
            add_bias = False
            self.bn = nn.BatchNorm1d(out_channels, momentum=0.05)
        else:
            add_bias = True

        # Linear operations
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=add_bias)

    def forward(self, x):
        batch_size, feat, length = x.shape
        assert self.in_channels == feat

        x = self.conv(x)

        if hasattr(self, "ln"):
            x = self.ln(x)
        elif hasattr(self, "bn"):
            x = self.bn(x)

        x = self.act(x)
        x = self.drop(x)
        return x

    # def load_wired_state_dict(self, wired_state_dict):
    #
    #     for k in wired_state_dict:
    #         if "ln.gamma" in k:
    #             if hasattr(self, "ln"):
    #                 raise NotImplementedError
    #         elif 'ln.gamma' in k:
    #             if hasattr(self, "ln"):
    #                 raise NotImplementedError
    #         elif 'ln.beta' in k:
    #             if hasattr(self, "ln"):
    #                 raise NotImplementedError
    #         elif 'bn.weight' in k:
    #             if hasattr(self, "bn"):
    #                 self.bn.weight.data = wired_state_dict[k]
    #         elif 'bn.bias' in k:
    #             if hasattr(self, "bn"):
    #                 self.bn.bias.data = wired_state_dict[k]
    #         elif 'bn.running_mean' in k:
    #             if hasattr(self, "bn"):
    #                 self.bn.running_mean = wired_state_dict[k]
    #         elif 'bn.running_var' in k:
    #             if hasattr(self, "bn"):
    #                 self.bn.running_var = wired_state_dict[k]
    #         elif 'bn.num_batches_tracked' in k:
    #             if hasattr(self, "bn"):
    #                 self.bn.num_batches_tracked = wired_state_dict[k]
    #         elif 'wx.weight' in k:
    #             _out, _in = wired_state_dict[k].shape
    #             _s2 = self.conv.weight.data.shape
    #             if _in == 440:
    #                 self.conv.weight.data = wired_state_dict[k].unsqueeze(1)
    #             else:
    #                 self.conv.weight.data = wired_state_dict[k].unsqueeze(2)
    #         elif 'wx.bias' in k:
    #             if self.conv.bias is not None:
    #                 self.conv.bias.data = wired_state_dict[k]
    #         else:
    #             raise ValueError


class TDNN(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, outputs,
                 layer_size,
                 dropout,
                 layer_batchnorm,
                 layer_layernorm,
                 activations):
        super(TDNN, self).__init__()
        assert len(layer_size) == len(dropout) == len(layer_batchnorm) == len(layer_layernorm) == len(activations), \
            (len(layer_size), len(dropout), len(layer_batchnorm), len(layer_layernorm), len(activations))
        self.input_feat_name = input_feat_name
        self.input_feat_length = input_feat_length
        self.context_left = 5
        self.context_right = 5

        self.layers = nn.ModuleList()

        self.layers.append(
            TDNNLayer(1, layer_size[0], activations[0], dropout[0], layer_layernorm[0], layer_batchnorm[0],
                      kernel_size=self.input_feat_length * (self.context_left + self.context_right + 1),
                      stride=self.input_feat_length))

        for i in range(1, len(layer_size)):
            # 1x1 conv
            self.layers.append(
                TDNNLayer(layer_size[i - 1], layer_size[i], activations[i], dropout[i], layer_layernorm[i],
                          layer_batchnorm[i]))

        self.output_layers = nn.ModuleDict({})
        self.out_names = []
        for _output_name, _output_num in outputs.items():
            self.out_names.append(_output_name)
            self.output_layers[_output_name] = TDNNLayer(layer_size[-1], _output_num, activation="log_softmax")

            self.batch_ordering = 'NCL'

    def info(self):
        return f" context: {self.context_left}, {self.context_right}" \
               + f" receptive_field: {self.context_left + self.context_right}"

    def forward(self, _input):
        x = _input[self.input_feat_name]
        batchsize, featlen, length = x.shape

        # ### TODO RM input transfor,
        # length, batchsize, featlen, context = x.shape
        # # NLCT
        # _input = torch.zeros((batchsize, 1, featlen * (length + context - 1)))
        #
        # for i in range(length):
        #     _input[:, 0, i * featlen:i * featlen + featlen] = x[i, :, :, 0]
        #
        # for c in range(1, context):
        #     _input[:, 0, (i + c) * featlen:(i + c) * featlen + featlen] = x[i, :, :, c]
        #
        # x = _input
        # ### /TODO RM input transfor,
        assert featlen * length % featlen * (self.context_right + self.context_left + 1) == 0
        x = x.view(batchsize, 1, featlen * length)

        for layer in self.layers:
            x = layer(x)

        out_length = x.shape[2]
        assert out_length == length - (self.context_left + self.context_right)

        out_dnn = x

        out_dict = {}
        for _output_name, _output_layers in self.output_layers.items():
            out_dict[_output_name] = _output_layers(out_dnn)

        return out_dict

    def load_warm_start(self, state_dict):
        checkpoint = torch.load(state_dict, map_location='cpu')

        for k in checkpoint:
            if 'output_layer' in k:
                new_t = self.state_dict()[k]
                new_t[1:] = checkpoint[k]
                checkpoint[k] = new_t

        self.load_state_dict(checkpoint)


# def load_wired_state_dict(self, wired_state_dict):
#     for i in range(5):
#         _wired_state_dict = {k: v for k, v in wired_state_dict.items() if k.startswith(f"MLP.layers.{i}.")}
#         self.layers[i].load_wired_state_dict(_wired_state_dict)
#
#     _wired_state_dict = {k: v for k, v in wired_state_dict.items() if
#                          k.startswith("output_layers.out_cd.layers.0.")}
#     self.output_layers['out_cd'].load_wired_state_dict(_wired_state_dict)


if __name__ == '__main__':
    model_path = "/mnt/data/pytorch-kaldi/trained_models/libri_MLP_ce/libri_TDNN_fbank_20190205_025557_better/checkpoints/checkpoint-epoch9.pth"

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict_orig = checkpoint['state_dict']
    state_dict = {k.replace("tdnn", "MLP").replace("MLP.layers.5.", "output_layers.out_cd.layers.0."): v for k, v in
                  state_dict_orig.items()}

    mlp = TDNN(input_feat_length=40,
               input_feat_name='fbank',
               outputs={'out_cd': 3480, "out_mono": 346},
               layer_size=[1024, 1024, 1024, 1024, 1024],
               dropout=[0.15, 0.15, 0.15, 0.15, 0.15],
               layer_batchnorm=[True, True, True, True, True],
               layer_layernorm=[False, False, False, False, False],
               activations=['relu', 'relu', 'relu', 'relu', 'relu'])
    # mlp = MLP(input_feat_length=40, input_feat_name='fbank', outputs={'out_cd': 3480, "out_mono": 346})

    _out_mono_layer_state_dict = {k: v for k, v in mlp.state_dict().items() if "out_mono" in k}
    state_dict.update(_out_mono_layer_state_dict)
    mlp.load_wired_state_dict(state_dict)
    # mlp_state_dict = mlp.load_state_dict(state_dict)

    test_input = torch.load("/mnt/data/pytorch-kaldi/pytorch-kaldi/mlp_saved_input.pth")
    mlp.eval()
    test_out = mlp(test_input)
    torch.save(mlp.state_dict(),
               "/mnt/data/pytorch-kaldi/trained_models/libri_MLP_ce/libri_TDNN_fbank_20190205_025557_better/checkpoints/checkpoint-epoch9_asTDNN.pth")

    print("t")
    # if 'dataset_sampler_state' not in checkpoint:
    #     checkpoint['dataset_sampler_state'] = None
    #
    # if checkpoint['dataset_sampler_state'] is None:
    #     start_epoch = checkpoint['epoch'] + 1
    # else:
    #     start_epoch = checkpoint['epoch']
    # global_step = checkpoint['global_step']
    # model.load_state_dict(checkpoint['state_dict'])
    #
    # assert (optimizers is None and lr_schedulers is None) \
    #        or (optimizers is not None and lr_schedulers is not None)
    # if optimizers is not None and lr_schedulers is not None:
    #     for opti_name in checkpoint['optimizers']:
    #         optimizers[opti_name].load_state_dict(checkpoint['optimizers'][opti_name])
    #     for lr_sched_name in checkpoint['lr_schedulers']:
    #         lr_schedulers[lr_sched_name].load_state_dict(checkpoint['lr_schedulers'][lr_sched_name])
    #
    # logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, start_epoch))
    # # TODO check checkpoint['dataset_sampler_state'] is none
    # return start_epoch, global_step, checkpoint['dataset_sampler_state']
