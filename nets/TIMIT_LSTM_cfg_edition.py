from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from base.base_model import BaseModel
from nets.modules.LSTM_cudnn import LSTM
from nets.modules.MLP import MLP


class TIMIT_LSTM(BaseModel):
    def __init__(self, inp_dim, lab_cd_num):
        super(TIMIT_LSTM, self).__init__()

        self.lstm = LSTM(inp_dim,
                         hidden_size=550,
                         num_layers=4,
                         bias=True,
                         batch_first=False,
                         dropout=0.2,
                         bidirectional=True)

        self.mlp_lab_cd = MLP(self.lstm.out_dim,
                              dnn_lay=[lab_cd_num],
                              dnn_drop=[0.0],
                              dnn_use_laynorm_inp=False,
                              dnn_use_batchnorm_inp=False,
                              dnn_use_batchnorm=[False],
                              dnn_use_laynorm=[False],
                              dnn_act=["softmax"])

        self.mlp_lab_mono = MLP(self.lstm.out_dim,
                                dnn_lay=[48],
                                dnn_drop=[0.0],
                                dnn_use_laynorm_inp=False,
                                dnn_use_batchnorm_inp=False,
                                dnn_use_batchnorm=[False],
                                dnn_use_laynorm=[False],
                                dnn_act=['softmax'])

        self.context_left = 0
        self.context_right = 0

    def forward(self, x):
        x = x['mfcc']
        out_dnn = self.lstm(x)
        if isinstance(out_dnn, PackedSequence):
            # Padd with zeros
            out_dnn, sequence_lengths = pad_packed_sequence(out_dnn)
        out_mono = self.mlp_lab_mono(out_dnn)
        out_cd = self.mlp_lab_cd(out_dnn)
        return {"out_cd": out_cd, "out_mono": out_mono}
