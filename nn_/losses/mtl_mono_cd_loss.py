import torch.nn.functional as F
from torch import nn


class MtlMonoCDLoss(nn.Module):

    def __init__(self, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono

    def forward(self, output, target):
        if len(target['lab_cd'].shape) == 2:
            _len = target['lab_cd'].shape[0]
            batch_size = target['lab_cd'].shape[1]

        elif len(target['lab_cd'].shape) == 1:
            _len = 1
            batch_size = target['lab_cd'].shape[0]
        else:
            raise ValueError

        num_cd = output['out_cd'].shape[-1]
        cd_max = target['lab_cd'].view(-1).max()

        assert cd_max < num_cd, "got max {}".format(cd_max)
        num_mono = output['out_mono'].shape[-1]
        mono_max = target['lab_mono'].view(-1).max()
        assert mono_max < num_mono, "got max {}".format(mono_max)

        loss_cd = F.nll_loss(output['out_cd'].view(_len * batch_size, -1),
                             target['lab_cd'].view(-1))
        loss_mono = F.nll_loss(output['out_mono'].view(_len * batch_size, -1),
                               target['lab_mono'].view(-1))

        loss_final = (self.weight_mono * loss_mono) + loss_cd
        return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
