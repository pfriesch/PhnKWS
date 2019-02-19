from base.base_model import BaseModel
from nn_.net_modules.CNN import CNN
from nn_.net_modules.MLP import MLP


class CNN_cd_mono_Net(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, lab_cd_num, lab_mono_num):
        super(CNN_cd_mono_Net, self).__init__()

        lab_cd_num += 2
        lab_mono_num += 2

        #
        self.cnn = CNN(input_feat_length,
                       N_filters=[80, 80, 80],
                       kernel_sizes=[3, 5, 6],
                       max_pool_len=[3, 0, 0],
                       activation=['relu', 'relu', 'relu'],
                       dropout=[0.15, 0.15, 0.0])

        context = self.cnn.receptive_field - 1
        lookahead = int(0.2 * context)

        self.context_left = context - lookahead
        self.context_right = lookahead

        self.mlp = MLP(self.cnn.N_out_feats,
                       dnn_lay=[1024],
                       dnn_drop=[0.10],
                       dnn_use_laynorm_inp=False,
                       dnn_use_batchnorm_inp=False,
                       dnn_use_batchnorm=[True],
                       dnn_use_laynorm=[False],
                       dnn_act=['relu'])

        self.mlp_lab_cd = MLP(self.mlp.out_dim,
                              dnn_lay=[lab_cd_num],
                              dnn_drop=[0.0],
                              dnn_use_laynorm_inp=False,
                              dnn_use_batchnorm_inp=False,
                              dnn_use_batchnorm=[False],
                              dnn_use_laynorm=[False],
                              dnn_act=['softmax'])

        self.mlp_lab_mono = MLP(self.mlp.out_dim,
                                dnn_lay=[lab_mono_num],
                                dnn_drop=[0.0],
                                dnn_use_laynorm_inp=False,
                                dnn_use_batchnorm_inp=False,
                                dnn_use_batchnorm=[False],
                                dnn_use_laynorm=[False],
                                dnn_act=['softmax'])

        self.input_feat_name = input_feat_name

    def forward(self, x):
        x = x[self.input_feat_name]
        out_dnn = self.cnn(x)

        max_len = x.shape[0]
        assert max_len == out_dnn.shape[0]
        batch_size = x.shape[1]
        assert batch_size == out_dnn.shape[1]
        out_dnn = out_dnn.view(max_len, batch_size, -1)

        out_mlp = self.mlp(out_dnn)

        out_cd = self.mlp_lab_cd(out_mlp)
        out_mono = self.mlp_lab_mono(out_mlp)
        return {'out_cd': out_cd, 'out_mono': out_mono}

    def get_sample_input(self):
        # TODO impl graph plotting wiht proper naming
        raise NotImplementedError
        # return torch.zeros((10, 5, 39))
