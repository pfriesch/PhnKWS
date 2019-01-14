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

        len = output['out_mono'].shape[0]
        batch_size = output['out_mono'].shape[1]

        pred = output['out_mono'].view(len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_mono'].view(-1)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy


class LabMonoError(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ignore_index = -1
        if isinstance(target["lab_mono"], PackedSequence):
            target['lab_mono'] = pad_packed_sequence(target['lab_mono'], padding_value=ignore_index)[0]

        len = output['out_mono'].shape[0]
        batch_size = output['out_mono'].shape[1]

        pred = output['out_mono'].view(len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_mono'].view(-1)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error


class LabCDAccuracy(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ignore_index = -1
        if isinstance(target["lab_cd"], PackedSequence):
            target['lab_cd'] = pad_packed_sequence(target['lab_cd'], padding_value=ignore_index)[0]

        len = output['out_cd'].shape[0]
        batch_size = output['out_cd'].shape[1]

        pred = output['out_cd'].view(len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_cd'].view(-1)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy


class LabCDError(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ignore_index = -1
        if isinstance(target["lab_cd"], PackedSequence):
            target['lab_cd'] = pad_packed_sequence(target['lab_cd'], padding_value=ignore_index)[0]

        len = output['out_cd'].shape[0]
        batch_size = output['out_cd'].shape[1]

        pred = output['out_cd'].view(len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_cd'].view(-1)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error
