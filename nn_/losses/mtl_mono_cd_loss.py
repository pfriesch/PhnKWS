import torch.nn.functional as F
from torch import nn


class MtlMonoCDLoss(nn.Module):

    def __init__(self, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono

    def forward(self, output, target):
        len = target['lab_cd'].shape[0]
        batch_size = target['lab_cd'].shape[1]

        num_cd = output['out_cd'].shape[2]
        cd_max = target['lab_cd'].view(-1).max()

        assert cd_max < num_cd, "got max {}".format(cd_max)
        num_mono = output['out_mono'].shape[2]
        mono_max = target['lab_mono'].view(-1).max()
        assert mono_max < num_mono, "got max {}".format(mono_max)

        loss_cd = F.nll_loss(output['out_cd'].view(len * batch_size, -1),
                             target['lab_cd'].view(-1))
        loss_mono = F.nll_loss(output['out_mono'].view(len * batch_size, -1),
                               target['lab_mono'].view(-1))

        loss_final = (self.weight_mono * loss_mono) + loss_cd
        return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
