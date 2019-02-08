import torch.nn.functional as F
from torch import nn

from data import PADDING_IGNORE_INDEX


class MonoLoss(nn.Module):

    def forward(self, output, target):
        if len(target['lab_mono'].shape) == 1:
            sequence_input = False
        elif len(target['lab_mono'].shape) == 2:
            sequence_input = True
        else:
            raise ValueError

        if sequence_input:
            seq_len = target['lab_mono'].shape[0]
            batch_size = target['lab_mono'].shape[1]
        elif not sequence_input:
            seq_len = 1
            batch_size = target['lab_mono'].shape[0]

        num_mono = output['out_mono'].shape[-1]
        mono_max = target['lab_mono'].view(-1).max()
        assert mono_max < num_mono, "got max {}".format(mono_max)

        loss_mono = F.nll_loss(output['out_mono'].view(seq_len * batch_size, -1),
                               target['lab_mono'].view(-1),
                               ignore_index=PADDING_IGNORE_INDEX)

        loss_final = loss_mono
        return {"loss_final": loss_final, "loss_mono": loss_mono}
