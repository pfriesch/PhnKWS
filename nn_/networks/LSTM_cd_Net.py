import torch

from base.base_model import BaseModel
from nn_.net_modules.LSTM_cudnn import LSTM
from nn_.net_modules.MLPModule import MLPModule


class LSTM_cd_Net(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, lab_cd_num):
        super(LSTM_cd_Net, self).__init__()

        self.lstm = LSTM(input_feat_length,
                         hidden_size=550,
                         num_layers=4,
                         bias=True,
                         batch_first=False,
                         dropout=0.2,
                         bidirectional=True)

        self.mlp_lab_cd = MLPModule(self.lstm.out_dim,
                                    dnn_lay=[lab_cd_num],
                                    dnn_drop=[0.0],
                                    dnn_use_laynorm_inp=False,
                                    dnn_use_batchnorm_inp=False,
                                    dnn_use_batchnorm=[False],
                                    dnn_use_laynorm=[False],
                                    dnn_act=["softmax"])

        self.context_left = 0
        self.context_right = 0
        self.input_feat_name = input_feat_name
        self.batch_ordering = "TNCL"
        # self.load_cfg()

    def forward(self, x):
        x = x[self.input_feat_name]
        out_dnn = self.lstm(x)
        # if isinstance(out_dnn, PackedSequence):
        #     # Padd with zeros
        #     out_dnn, sequence_lengths = pad_packed_sequence(out_dnn)

        max_len = x.shape[0]
        batch_size = x.shape[1]
        out_dnn = out_dnn.view(max_len * batch_size, -1)

        out_cd = self.mlp_lab_cd(out_dnn)
        out_mono = self.mlp_lab_mono(out_dnn)
        return {"out_cd": out_cd, "out_mono": out_mono}

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
        self.mlp_lab_cd.load_cfg()

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
