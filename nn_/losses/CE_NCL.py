import torch.nn.functional as F
from torch import nn

from data import PADDING_IGNORE_INDEX


class CELoss_NCL(nn.Module):

    def __init__(self, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono

    def forward(self, output, target):
        assert len(target['lab_cd'].shape) == 2

        # batch_size = output['out_cd'].shape[0]
        # seq_len = output['out_cd'].shape[2]

        num_cd = output['out_cd'].shape[1]
        cd_max = target['lab_cd'].view(-1).max()
        assert cd_max < num_cd, f"got max {cd_max}, expeced {num_cd} (min: {target['lab_cd'].view(-1).min()})"

        num_mono = output['out_mono'].shape[1]
        mono_max = target['lab_mono'].view(-1).max()
        assert mono_max < num_mono, f"got max {mono_max}, expeced {num_mono} (min: {target['lab_mono'].view(-1).min()})"

        loss_cd = F.nll_loss(output['out_cd'],
                             target['lab_cd'],
                             ignore_index=PADDING_IGNORE_INDEX)
        loss_mono = F.nll_loss(output['out_mono'],
                               target['lab_mono'],
                               ignore_index=PADDING_IGNORE_INDEX)

        loss_final = (self.weight_mono * loss_mono) + loss_cd
        return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
