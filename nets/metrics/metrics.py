import torch
from torch.nn import Module
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class LabMonoAccuracy(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ignore_index = -1
        if isinstance(target["lab_mono"], PackedSequence):
            target['lab_mono'] = pad_packed_sequence(target['lab_mono'], padding_value=ignore_index)[0]

        num_mono_labs = output['out_mono'].shape[2]

        accuracy = torch.mean((output['out_mono'].view(-1, num_mono_labs).max(dim=1)[1] ==
                               target['lab_mono'].view(-1)).to(dtype=torch.float32))

        return accuracy


class LabCDAccuracy(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ignore_index = -1
        if isinstance(target["lab_cd"], PackedSequence):
            target['lab_cd'] = pad_packed_sequence(target['lab_cd'], padding_value=ignore_index)[0]

        num_mono_labs = output['out_cd'].shape[2]

        accuracy = torch.mean((output['out_cd'].view(-1, num_mono_labs).max(dim=1)[1] ==
                               target['lab_cd'].view(-1)).to(dtype=torch.float32))

        return accuracy
