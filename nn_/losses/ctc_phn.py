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
    assert targets.device.type == "cpu", "CuDNN expects targets to on the CPU"
    assert target_sequence_lengths.device.type == "cpu", "CuDNN expects target_sequence_lengths to on the CPU"
    assert input_sequence_lengths.device.type == "cpu", "CuDNN expects input_sequence_lengths to on the CPU"


class CTCPhnLoss(nn.Module):

    def __init__(self, batch_ordering):
        super().__init__()
        self.batch_ordering = batch_ordering

    def forward(self, output, target):
        if self.batch_ordering == 'NCT' or self.batch_ordering == 'NCL':
            # NCT (NCL) -> TNC required for CuDNN
            logits = output['out_phn'].permute(2, 0, 1)
        elif self.batch_ordering == 'TNCL':
            assert len(output['out_phn'].shape) == 3
            logits = output['out_phn']
        else:
            raise NotImplementedError

        _targets = target['lab_phn']
        assert _targets.min() >= 0
        assert _targets.max() <= logits.shape[2]
        # all input_lengths must be T for CuDNN
        input_sequence_lengths = torch.full_like(
            target['input_sequence_lengths'], dtype=torch.int32, fill_value=logits.shape[0])
        target_sequence_lengths = target['target_sequence_lengths']

        check_cudnn_cond(_targets, target_sequence_lengths, input_sequence_lengths)

        loss_phn = F.ctc_loss(logits, _targets,
                              input_sequence_lengths, target_sequence_lengths)

        return {"loss_final": loss_phn}
