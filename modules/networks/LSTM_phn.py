from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from base.base_model import BaseModel
from modules.net_modules.LSTM_cudnn import LSTM
from modules.net_modules.MLP import MLP


class LSTM_phn(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, lab_phn_num):
        super(LSTM_phn, self).__init__()

        # self.lstm = LSTM(input_feat_length,
        #                  hidden_size=550,
        #                  num_layers=4,
        #                  bias=True,
        #                  batch_first=False,
        #                  dropout=0.2,
        #                  bidirectional=True)

        self.lstm = LSTM(input_feat_length,
                         hidden_size=50,
                         num_layers=2,
                         bias=True,
                         batch_first=False,
                         dropout=0.2,
                         bidirectional=True)

        self.mlp_phn = MLP(self.lstm.out_dim,
                           dnn_lay=[lab_phn_num],
                           dnn_drop=[0.0],
                           dnn_use_laynorm_inp=False,
                           dnn_use_batchnorm_inp=False,
                           dnn_use_batchnorm=[False],
                           dnn_use_laynorm=[False],
                           dnn_act=['log_softmax'])  # log softmax for ctc loss

        self.context_left = 0
        self.context_right = 0
        self.input_feat_name = input_feat_name

    def forward(self, x):
        x = x[self.input_feat_name]
        assert isinstance(x, PackedSequence)

        out_dnn = self.lstm(x)
        # Padd with zeros
        out_dnn, sequence_lengths = pad_packed_sequence(out_dnn)

        # max_len = out_dnn.shape[0]
        # batch_size = out_dnn.shape[1]
        # out_dnn = out_dnn.view(max_len * batch_size, -1)
        out_phn = self.mlp_phn(out_dnn)

        # mask = torch.ones_like(out_phn)
        # for i, _len in enumerate(sequence_lengths):
        #     mask[_len:, i] = 0
        # mask = mask.view(max_len * batch_size, -1)

        # out_phn_masked = out_phn
        return {"out_phn": out_phn, "sequence_lengths": sequence_lengths}

    def get_sample_input(self):
        # TODO impl graph plotting wiht proper naming
        raise NotImplementedError
        # return torch.zeros((10, 5, 39))
