from base.base_model import BaseModel
from _nn.net_modules.CNN import CNN
from _nn.net_modules.MLP import MLP


class CNN_cd_mono(BaseModel):
    def __init__(self, input_feat_length, context_left, context_right, input_feat_name, lab_cd_num, lab_mono_num):
        super(CNN_cd_mono, self).__init__()

        self.context_left = context_left
        self.context_right = context_right
        lab_cd_num += 2
        lab_mono_num += 2

        #
        self.cnn = CNN(input_feat_length,
                       # self.context_left + self.context_right,
                       N_filters=[80, 60, 60],
                       kernel_sizes=[10, 3, 3],
                       max_pool_len=[3, 2, 1],
                       use_laynorm=[True, True, True],
                       use_batchnorm=[False, False, False],
                       activation=['relu', 'relu', 'relu'],
                       dropout=[0.15, 0.15, 0.15])
        #
        # self.mlp = MLP(self.cnn.out_dim,
        #                dnn_lay=[1024, 1024, 1024, 1024],
        #                dnn_drop=[0.10, 0.10, 0.10, 0.10],
        #                dnn_use_laynorm_inp=False,
        #                dnn_use_batchnorm_inp=False,
        #                dnn_use_batchnorm=[True, True, True, True],
        #                dnn_use_laynorm=[False, False, False, False],
        #                dnn_act=['relu', 'relu', 'relu', 'relu'])

        # self.cnn = CNN(input_feat_length,
        #                self.context_left + self.context_right,
        #                N_filters=[80, 60],
        #                kernel_sizes=[10, 3],
        #                max_pool_len=[3, 1],
        #                use_laynorm_inp=False,
        #                use_batchnorm_inp=False,
        #                use_laynorm=[True, True],
        #                use_batchnorm=[False, False],
        #                activation=['relu', 'relu'],
        #                dropout=[0.15, 0.15])
        #
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
        # model_proto = proto / model.proto
        # model = out_dnn1 = compute(CNN_layers, raw)
        # out_dnn2 = compute(MLP_layers, out_dnn1)
        # out_dnn3 = compute(MLP_soft1, out_dnn2)
        # out_dnn4 = compute(MLP_soft2, out_dnn2)
        # loss_mono = cost_nll(out_dnn4, lab_mono)
        # loss_mono_w = mult_constant(loss_mono, 1.0)
        # loss_cd = cost_nll(out_dnn3, lab_cd)
        # loss_final = sum(loss_cd, loss_mono_w)
        # err_final = cost_err(out_dnn3, lab_cd)

        x = x[self.input_feat_name]
        out_dnn = self.cnn(x)
        # if isinstance(out_dnn, PackedSequence):
        #     # Padd with zeros
        #     out_dnn, sequence_lengths = pad_packed_sequence(out_dnn)

        max_len = x.shape[0]
        assert max_len == out_dnn.shape[0]
        batch_size = x.shape[1]
        assert batch_size == out_dnn.shape[1]
        out_dnn = out_dnn.reshape(max_len, batch_size, -1)
        # out_dnn = out_dnn.reshape(batch * seq_len, -1)

        out_mlp = self.mlp(out_dnn)

        out_cd = self.mlp_lab_cd(out_mlp)
        out_mono = self.mlp_lab_mono(out_mlp)
        return {'out_cd': out_cd, 'out_mono': out_mono}

    def get_sample_input(self):
        # TODO impl graph plotting wiht proper naming
        raise NotImplementedError
        # return torch.zeros((10, 5, 39))

    # def load_cfg(self):
    #     nns = torch.load('/mnt/data/pytorch-kaldi_cfg/nns.pyt')
    #
    #     _lstm = {k.replace('lstm.0.', 'lstm.'): nns['LSTM_cudnn_layers'][k] for k in nns['LSTM_cudnn_layers']}
    #     _MLP_layers = {k: nns['MLP_layers'][k] for k in nns['MLP_layers']}
    #     _MLP_layers2 = {k: nns['MLP_layers2'][k] for k in nns['MLP_layers2']}
    #     # curr_state = self.lstm.state_dict()
    #     self.lstm.load_state_dict(_lstm)
    #     self.mlp_lab_cd.load_state_dict(_MLP_layers)
    #     self.mlp_lab_mono.load_state_dict(_MLP_layers2)
    #
    #     print('Done Loading')
