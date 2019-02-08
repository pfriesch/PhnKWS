from base.base_model import BaseModel

from nn_.net_modules.MLP import MLP


class TDNN_mono(BaseModel):
    def __init__(self, input_feat_length, input_feat_name, lab_mono_num):
        super(TDNN_mono, self).__init__()
        self.input_feat_name = input_feat_name
        self.input_feat_length = input_feat_length
        self.context_left = 5
        self.context_right = 5

        self.tdnn = MLP(input_feat_length * (self.context_left + self.context_right + 1),
                        dnn_lay=[1024, 1024, 1024, 1024, 1024, lab_mono_num],
                        dnn_drop=[0.15, 0.15, 0.15, 0.15, 0.15, 0.0],
                        dnn_use_laynorm_inp=False,
                        dnn_use_batchnorm_inp=False,
                        dnn_use_batchnorm=[True, True, True, True, True, False],
                        dnn_use_laynorm=[False, False, False, False, False, False],
                        dnn_act=['relu', 'relu', 'relu', 'relu', 'relu', 'log_softmax'])

    def forward(self, x):
        x = x[self.input_feat_name]

        if len(x.shape) == 3:
            sequence_input = False
        elif len(x.shape) == 4:
            sequence_input = True
        else:
            raise ValueError

        if sequence_input:
            T = x.shape[0]
            batch = x.shape[1]
            feats = x.shape[2]
            assert feats == self.input_feat_length
            context = x.shape[3]
            assert context == self.context_left + self.context_right + 1
            x = x.view(T * batch, feats * context)
        elif not sequence_input:
            batch = x.shape[0]
            feats = x.shape[1]
            assert feats == self.input_feat_length
            context = x.shape[2]
            assert context == self.context_left + self.context_right + 1
            x = x.view(batch, feats * context)
        else:
            raise ValueError

        out_dnn = self.tdnn(x)

        if sequence_input:
            out_mono = out_dnn.view(T, batch, -1)
        elif not sequence_input:
            out_mono = out_dnn.view(batch, -1)
        else:
            raise ValueError

        return {'out_mono': out_mono}
