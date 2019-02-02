from base.base_model import BaseModel

from nn_.net_modules.CNN_cfg import CNN
from nn_.net_modules.MLP import MLP


class CNN_cd(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, lab_cd_num):
        super(CNN_cd, self).__init__()
        self.input_feat_name = input_feat_name
        self.input_feat_length = input_feat_length
        self.context_left = 5
        self.context_right = 5

        self.cnn = CNN(input_feat_length * (self.context_right + self.context_left + 1),
                       input_context=self.context_left + self.context_right,
                       N_filters=[80, 60, 60],
                       kernel_sizes=[10, 3, 3],
                       max_pool_len=[3, 2, 1],
                       use_laynorm_inp=False,
                       use_batchnorm_inp=False,
                       use_laynorm=[True, True, True],
                       use_batchnorm=[False, False, False],
                       activation=['relu', 'relu', 'relu'],
                       dropout=[0.15, 0.15, 0.15], )

        self.mlp_lab_cd = MLP(self.cnn.out_dim,
                              dnn_lay=[1024, 1024, 1024, 1024, lab_cd_num],
                              dnn_drop=[0.15, 0.15, 0.15, 0.15, 0.0],
                              dnn_use_laynorm_inp=False,
                              dnn_use_batchnorm_inp=False,
                              dnn_use_batchnorm=[True, True, True, True, False],
                              dnn_use_laynorm=[False, False, False, False, False],
                              dnn_act=['relu', 'relu', 'relu', 'relu', 'log_softmax'])

    def forward(self, x):
        x = x[self.input_feat_name]
        T = x.shape[0]
        batch = x.shape[1]
        feats = x.shape[2]
        assert feats == self.input_feat_length
        context = x.shape[3]
        assert context == self.context_left + self.context_right + 1
        x = x.view(T * batch, feats * context).unsqueeze(1)

        out_dnn = self.cnn(x)

        out_dnn = out_dnn.view(T, batch, -1)

        out_cd = self.mlp_lab_cd(out_dnn)
        return {'out_cd': out_cd}
