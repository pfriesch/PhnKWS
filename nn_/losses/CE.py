import torch.nn.functional as F
from torch import nn

from data import PADDING_IGNORE_INDEX


class CELoss(nn.Module):

    def __init__(self, batch_ordering, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono
        self.batch_ordering = batch_ordering

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

        if 'lab_phnframe' in target:
            num_phnframe = output['out_phnframe'].shape[1]
            phnframe_max = target['lab_phnframe'].view(-1).max()
            assert phnframe_max < num_phnframe, \
                f"got max {phnframe_max}," \
                + f" expeced {num_phnframe} (min: {target['lab_phnframe'].view(-1).min()})"

        loss_cd = F.nll_loss(output['out_cd'],
                             target['lab_cd'],
                             ignore_index=PADDING_IGNORE_INDEX)
        loss_mono = F.nll_loss(output['out_mono'],
                               target['lab_mono'],
                               ignore_index=PADDING_IGNORE_INDEX)

        if 'lab_phnframe' in target:
            loss_phnframe = F.nll_loss(output['out_phnframe'],
                                       target['lab_phnframe'],
                                       ignore_index=PADDING_IGNORE_INDEX)

            loss_final = (self.weight_mono * loss_mono) + loss_cd + loss_phnframe
            return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono,
                    "loss_phnframe": loss_phnframe}

        else:

            loss_final = (self.weight_mono * loss_mono) + loss_cd
            return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
