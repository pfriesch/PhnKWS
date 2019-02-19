from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from base.base_model import BaseModel
from nn_.net_modules.GCNN import GCNN


class GCNN_cd_mono_Net(BaseModel):

    def __init__(self, num_input_feats,
                 input_feat_name,
                 lab_cd_num,
                 lab_mono_num,
                 N_filters,
                 kernel_sizes,
                 dropout):
        super(GCNN_cd_mono_Net, self).__init__()
        self.input_feat_name = input_feat_name
        self.num_input_feats = num_input_feats

        gcnn_out_num = 500

        self.gcnn = GCNN(num_input_feats=self.num_input_feats,
                         num_output_feats=gcnn_out_num,
                         N_filters=N_filters,
                         kernel_sizes=kernel_sizes,
                         dropout=dropout[:-1])

        self.linear_lab_cd = weight_norm(nn.Linear(gcnn_out_num, lab_cd_num), dim=0)
        self.linear_lab_mono = weight_norm(nn.Linear(gcnn_out_num, lab_mono_num), dim=0)

        self.context_left, self.context_right = self.context()

    def forward(self, x):
        x = x[self.input_feat_name]
        x = x.permute(1, 2, 0).unsqueeze(2)

        x = self.gcnn(x)

        out_cd = F.log_softmax(self.linear_lab_cd(x), dim=3)
        out_mono = F.log_softmax(self.linear_lab_mono(x), dim=3)
        return {'out_cd': out_cd, 'out_mono': out_mono}

    def context(self):
        return self.gcnn.context()
