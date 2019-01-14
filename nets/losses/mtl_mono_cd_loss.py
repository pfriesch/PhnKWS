import torch
import torch.nn.functional as F
from torch.nn import NLLLoss, Module
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class MtlMonoCDLoss(Module):

    def __init__(self, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono

    def forward(self, output, target):
        ignore_index = -1
        if any([isinstance(target[lab], PackedSequence) for lab in target]):
            target = {lab: pad_packed_sequence(target[lab], padding_value=ignore_index)[0] for lab in target}

        len = output['out_cd']
        batch_size = output['out_cd'].shape[1]

        loss_cd = F.nll_loss(output['out_cd'].view(len * batch_size, -1),
                             target['lab_cd'].view(-1),
                             ignore_index=ignore_index)
        loss_mono = F.nll_loss(output['out_mono'].view(len * batch_size, -1),
                               target['lab_mono'].view(-1),
                               ignore_index=ignore_index)

        loss_final = loss_cd + (self.weight_mono * loss_mono)
        return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
