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
