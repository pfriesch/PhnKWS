import torch

import torch.nn.functional as F
from torch import nn


def check_cudnn_cond(targets, target_sequence_lengths, input_sequence_lengths):
    assert len(targets.shape) == 1, "CuDNN expects concatinated targets"
    assert (targets == 0).sum() == 0, "CuDNN expects blank 0"
    assert all(input_sequence_lengths == input_sequence_lengths[0]), \
        "CuDNN expects input_sequence_lengths to be all of the same length"
    assert all(target_sequence_lengths <= 256), "CuDNN expects targets shorter or equal to 256"
    assert targets.dtype == torch.int32, "CuDNN expects targets to be in32"


class CTCPhnLoss(nn.Module):

    def forward(self, output, target):
        logits = output['out_phn']
        _targets = target['lab_phn']
        input_sequence_lengths = torch.full_like(
            target['input_sequence_lengths'], dtype=torch.int32, fill_value=logits.shape[0])
        target_sequence_lengths = target['target_sequence_lengths']

        check_cudnn_cond(_targets, target_sequence_lengths, input_sequence_lengths)

        loss_phn = F.ctc_loss(logits, _targets,
                              input_sequence_lengths, target_sequence_lengths)

        return {"loss_final": loss_phn}
