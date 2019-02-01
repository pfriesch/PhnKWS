from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from modules.net_modules.GCNN import GCNN
from base.base_model import BaseModel


class GCNN_phn(BaseModel):

    def __init__(self, num_input_feats,
                 input_feat_name,
                 lab_phn_num,
                 N_filters,
                 kernel_sizes,
                 dropout):
        super(GCNN_phn, self).__init__()
        self.input_feat_name = input_feat_name
        self.num_input_feats = num_input_feats

        gcnn_out_num = 500

        self.gcnn = GCNN(num_input_feats=self.num_input_feats,
                         num_output_feats=gcnn_out_num,
                         N_filters=N_filters,
                         kernel_sizes=kernel_sizes,
                         dropout=dropout[:-1])

        self.linear_phn = weight_norm(nn.Linear(gcnn_out_num, lab_phn_num), dim=0)

        self.context_left = sum(self.context())
        self.context_right = 0

    def forward(self, in_dict):
        x = in_dict[self.input_feat_name]["input"]
        x = x.permute(1, 2, 0).unsqueeze(2)
        # [B, C, H, W]

        x = self.gcnn(x)

        out_phn = F.log_softmax(self.linear_phn(x), dim=3)
        return {'out_phn': out_phn, "sequence_lengths": in_dict[self.input_feat_name][
                                                            "sequence_lengths"] - self.context_left - self.context_right}

    def context(self):
        return self.gcnn.context()
