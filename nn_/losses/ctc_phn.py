import torch.nn.functional as F
from torch import nn


class CTCPhnLoss(nn.Module):

    def forward(self, output, target):
        assert len((target['lab_phn']['label'][0] == 0).nonzero()) == 0  # blank symbol is 0
        logits = output['out_phn'].squeeze(1).permute(1, 0, 2)
        target_concat = target['lab_phn']['label'][0]
        target_sequence_lengths = target['lab_phn']['sequence_lengths']
        loss_phn = F.ctc_loss(logits, target_concat,
                              output['sequence_lengths'], target_sequence_lengths)

        return {"loss_final": loss_phn}
