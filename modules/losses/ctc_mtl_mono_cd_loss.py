import torch.nn.functional as F
from torch import nn


class CTCMtlMonoCDLoss(nn.Module):

    def __init__(self, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono

    def forward(self, output, target):
        loss_cd = F.ctc_loss(output['out_cd'],
                             target['lab_cd'])
        loss_mono = F.ctc_loss(output['out_mono'],
                               target['lab_mono'])

        loss_final = (self.weight_mono * loss_mono) + loss_cd
        return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
