import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class CDLoss(Module):

    def forward(self, output, target):
        ignore_index = -1
        if any([isinstance(target[lab], PackedSequence) for lab in target]):
            target = {lab: pad_packed_sequence(target[lab], padding_value=ignore_index)[0] for lab in target}

        len = output['out_cd'].shape[0]
        batch_size = output['out_cd'].shape[1]

        loss_final = F.nll_loss(output['out_cd'].view(len * batch_size, -1),
                                target['lab_cd'].view(-1),
                                ignore_index=ignore_index)

        return {"loss_final": loss_final, "loss_cd": loss_final}
