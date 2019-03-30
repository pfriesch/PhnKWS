import torch

from torch import nn

from base.base_model import BaseModel
from nn_.net_modules.LSTM_cudnn import LSTM
from nn_.net_modules.MLPModule import MLPModule


class LSTMNet(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, outputs,
                 hidden_size=550, num_layers=4, bias=True, dropout=0.2, bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = LSTM(input_feat_length,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         bias=bias,
                         batch_first=False,
                         dropout=dropout,
                         bidirectional=bidirectional)

        self.output_layers = nn.ModuleDict({})
        self.out_names = []
        for _output_name, _output_num in outputs.items():
            self.out_names.append(_output_name)

            self.output_layers[_output_name] = nn.ModuleList([
                MLPModule(self.lstm.out_dim,
                          dnn_lay=[_output_num],
                          dnn_drop=[0.0],
                          dnn_use_laynorm_inp=False,
                          dnn_use_batchnorm_inp=False,
                          dnn_use_batchnorm=[False],
                          dnn_use_laynorm=[False],
                          dnn_act=['log_softmax'])
            ])

        self.context_left = 0
        self.context_right = 0
        self.input_feat_name = input_feat_name
        self.batch_ordering = "TNCL"
        self.load_cfg()

    def forward(self, x):
        x = x[self.input_feat_name]
        x = x.squeeze(3)
        seq_len, batch, feats = x.shape
        out_dnn = self.lstm(x)

        # mask = torch.ones_like(out_phn)
        # for i, _len in enumerate(sequence_lengths):
        #     mask[_len:, i] = 0
        # mask = mask.view(max_len * batch_size, -1)

        # out_phn_masked = out_phn
        out_dict = {}
        for _output_name, _output_layers in self.output_layers.items():
            out_dnn = out_dnn.view(seq_len * batch, -1)
            out_dict[_output_name] = _output_layers[0](out_dnn)
            out_dict[_output_name] = out_dict[_output_name].view(seq_len, batch, -1)

        return out_dict

    def get_sample_input(self):
        # TODO impl graph plotting wiht proper naming
        raise NotImplementedError
        # return torch.zeros((10, 5, 39))

    def load_cfg(self):
        final_architecture1 = torch.load(
            "/mnt/data/pytorch-kaldi_cfg/exp/libri_LSTM_fbank/exp_files/final_architecture1.pkl", map_location='cpu')[
            'model_par']
        final_architecture1 = {k.replace('lstm.0.', 'lstm.'): v for k, v in final_architecture1.items()}

        self.lstm.load_state_dict(final_architecture1)

        final_architecture2 = torch.load(
            "/mnt/data/pytorch-kaldi_cfg/exp/libri_LSTM_fbank/exp_files/final_architecture2.pkl",
            map_location='cpu')[
            'model_par']

        final_architecture2 = {'out_cd.0.layers.0.' + k.replace('0.', ''): v for k, v in final_architecture2.items()}

        # _statedict = self.output_layers.state_dict()
        # for k in _statedict:
        #     print(k)
        #     # if not k == "out_cd.0.layers.0.bn.num_batches_tracked":
        #         _statedict[k][1:] = final_architecture2[k]

        self.output_layers.load_state_dict(final_architecture2)

        # self.output_layers.load_state_dict(final_architecture2)

        # 'wx.0.weight' (139634859952944)
        # 'out_cd.0.layers.0.ln.gamma' (139634859745808)

        # final_architecture2 = {k: v for k, v in final_architecture2.items()}

        # _MLP_layers = {k: final_architecture2['MLP_layers'][k] for k in final_architecture2['MLP_layers']}
        # curr_state = self.lstm.state_dict()
        # self.lstm.load_state_dict(_lstm)
        # self.mlp_lab_cd.load_state_dict(_MLP_layers)
        # self.mlp_lab_mono.load_state_dict(_MLP_layers2)

        print("Done Loading")

    # def load_cfg(self):
    #     nns = torch.load("/mnt/data/pytorch-kaldi_cfg/nns.pyt")
    #
    #     _lstm = {k.replace("lstm.0.", "lstm."): nns['LSTM_cudnn_layers'][k] for k in nns['LSTM_cudnn_layers']}
    #     _MLP_layers = {k: nns['MLP_layers'][k] for k in nns['MLP_layers']}
    #     _MLP_layers2 = {k: nns['MLP_layers2'][k] for k in nns['MLP_layers2']}
    #     # curr_state = self.lstm.state_dict()
    #     self.lstm.load_state_dict(_lstm)
    #     self.mlp_lab_cd.load_state_dict(_MLP_layers)
    #     self.mlp_lab_mono.load_state_dict(_MLP_layers2)
    #
    #     print("Done Loading")
