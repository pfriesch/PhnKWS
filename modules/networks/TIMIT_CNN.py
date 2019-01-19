import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from modules.net_modules.CNN import CNN
from modules.net_modules.MLP import MLP


class TIMIT_CNN(nn.Module):
    def __init__(self, inp_dim, lab_cd_num):
        super(TIMIT_CNN, self).__init__()

        self.cnn = CNN(inp_dim,
                       N_filters=[80, 60, 60],
                       len_filters=[10, 3, 3],
                       max_pool_len=[3, 2, 1],
                       use_laynorm_inp=False,
                       use_batchnorm_inp=False,
                       use_laynorm=[True, True, True],
                       use_batchnorm=[False, False, False],
                       activation=['relu', 'relu', 'relu'],
                       dropout=[0.15, 0.15, 0.15])

        self.mlp_lab_cd = MLP(self.lstm.out_dim,
                              dnn_lay=[1024, 1024, 1024, 1024, lab_cd_num],
                              dnn_drop=[0.15, 0.15, 0.15, 0.15, 0.0],
                              dnn_use_laynorm_inp=False,
                              dnn_use_batchnorm_inp=False,
                              dnn_use_batchnorm=[True, True, True, True, False],
                              dnn_use_laynorm=[False, False, False, False, False],
                              dnn_act=["relu", "relu", "relu", "relu", "softmax"])

        self.context_left = 0
        self.context_right = 0

    def forward(self, x):
        x = x['fbank']
        out_dnn = self.cnn(x)
        if isinstance(out_dnn, PackedSequence):
            # Padd with zeros
            out_dnn, sequence_lengths = pad_packed_sequence(out_dnn)

        max_len = x.shape[0]
        batch_size = x.shape[1]
        out_dnn = out_dnn.view(max_len * batch_size, -1)

        out_cd = self.mlp_lab_cd(out_dnn)
        out_mono = self.mlp_lab_mono(out_dnn)
        return {"out_cd": out_cd, "out_mono": out_mono}
