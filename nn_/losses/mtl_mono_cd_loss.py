import torch.nn.functional as F
from torch import nn

from data import PADDING_IGNORE_INDEX


class MtlMonoCDLoss(nn.Module):

    def __init__(self, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono

    def forward(self, output, target):
        if len(target['lab_cd'].shape) == 1:
            sequence_input = False
        elif len(target['lab_cd'].shape) == 2:
            sequence_input = True
        else:
            raise ValueError

        if sequence_input:
            batch_size = target['lab_cd'].shape[0]
            seq_len = target['lab_cd'].shape[1]
        elif not sequence_input:
            batch_size = target['lab_cd'].shape[0]
            seq_len = 1

        num_cd = output['out_cd'].shape[1]
        cd_max = target['lab_cd'].view(-1).max()

        assert cd_max < num_cd, f"got max {cd_max}, expected {num_cd}"
        num_mono = output['out_mono'].shape[1]
        mono_max = target['lab_mono'].view(-1).max()
        assert mono_max < num_mono, f"got max {mono_max}, expected {num_mono} " \
                                    + f"(min: {target['lab_mono'].view(-1).min()})"

        # loss_cd = F.nll_loss(output['out_cd'].permute(0, 2, 1).contiguous().view(seq_len * batch_size, -1),
        loss_cd = F.nll_loss(output['out_cd'].view(seq_len * batch_size, -1),
                             target['lab_cd'].view(-1),
                             ignore_index=PADDING_IGNORE_INDEX)
        # loss_mono = F.nll_loss(output['out_mono'].permute(0, 2, 1).contiguous().view(seq_len * batch_size, -1),
        loss_mono = F.nll_loss(output['out_mono'].view(seq_len * batch_size, -1),
                               target['lab_mono'].view(-1),
                               ignore_index=PADDING_IGNORE_INDEX)

        loss_final = (self.weight_mono * loss_mono) + loss_cd
        return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
