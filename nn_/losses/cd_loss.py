import torch.nn.functional as F
from torch import nn


class CDLoss(nn.Module):

    def forward(self, output, target):
        ignore_index = 0
        len = output['out_cd'].shape[0]
        batch_size = output['out_cd'].shape[1]

        loss_final = F.nll_loss(output['out_cd'].view(len * batch_size, -1),
                                target['lab_cd'].view(-1),
                                ignore_index=ignore_index)

        return {"loss_final": loss_final, "loss_cd": loss_final}
