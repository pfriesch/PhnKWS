import torch.nn as nn

from _nn.net_modules.MLP import MLP


class TIMIT_MLP(nn.Module):
    def __init__(self, inp_dim, lab_cd_num):
        super(TIMIT_MLP, self).__init__()

        self.mlp = MLP(inp_dim,
                       dnn_lay=[1024, 1024, 1024, 1024, 1024],
                       dnn_drop=[0.15, 0.15, 0.15, 0.15, 0.15],
                       dnn_use_laynorm_inp=False,
                       dnn_use_batchnorm_inp=False,
                       dnn_use_batchnorm=[True, True, True, True, True],
                       dnn_use_laynorm=[False, False, False, False, False],
                       dnn_act=["relu", "relu", "relu", "relu", "relu"])

        self.mlp_lab_cd = MLP(inp_dim,
                              dnn_lay=[lab_cd_num],
                              dnn_drop=[0.0],
                              dnn_use_laynorm_inp=False,
                              dnn_use_batchnorm_inp=False,
                              dnn_use_batchnorm=[False],
                              dnn_use_laynorm=[False],
                              dnn_act=["softmax"])

        self.mlp_lab_mono = MLP(inp_dim,
                                dnn_lay=[48],
                                dnn_drop=[0.0],
                                dnn_use_laynorm_inp=False,
                                dnn_use_batchnorm_inp=False,
                                dnn_use_batchnorm=[False],
                                dnn_use_laynorm=[False],
                                dnn_act=["softmax"])

        self.context_left = 0
        self.context_right = 0

    def forward(self, x):
        x = x[self.input_feat_name]

        max_len = x.shape[0]
        batch_size = x.shape[1]
        x = x.view(max_len * batch_size, -1)

        out_dnn = self.mlp(x)
        out_cd = self.mlp_lab_cd(out_dnn)
        out_mono = self.mlp_lab_mono(out_dnn)
        return {"out_cd": out_cd, "out_mono": out_mono}
