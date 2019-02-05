import torch.nn.functional as F
from torch import nn

from data import PADDING_IGNORE_INDEX


class CDLoss(nn.Module):

    def forward(self, output, target):
        if len(target['lab_cd'].shape) == 1:
            sequence_input = False
        elif len(target['lab_cd'].shape) == 2:
            sequence_input = True
        else:
            raise ValueError

        if sequence_input:
            seq_len = target['lab_cd'].shape[0]
            batch_size = target['lab_cd'].shape[1]
        elif not sequence_input:
            seq_len = 1
            batch_size = target['lab_cd'].shape[0]

        num_cd = output['out_cd'].shape[-1]
        cd_max = target['lab_cd'].view(-1).max()
        assert cd_max < num_cd, "got max {}".format(cd_max)

        loss_cd = F.nll_loss(output['out_cd'].view(seq_len * batch_size, -1),
                             target['lab_cd'].view(-1),
                             ignore_index=PADDING_IGNORE_INDEX)
        loss_final = loss_cd
        return {"loss_final": loss_final, "loss_cd": loss_final}
