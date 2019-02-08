import torch
from torch.nn import Module


class LabMonoAccuracy(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        if len(target['lab_mono'].shape) == 2:
            _len = target['lab_mono'].shape[0]
            batch_size = target['lab_mono'].shape[1]

        elif len(target['lab_mono'].shape) == 1:
            _len = 1
            batch_size = target['lab_mono'].shape[0]
        else:
            raise ValueError

        pred = output['out_mono'].view(_len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_mono'].view(-1)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy.item()


class LabMonoError(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        if len(target['lab_mono'].shape) == 2:
            _len = target['lab_mono'].shape[0]
            batch_size = target['lab_mono'].shape[1]

        elif len(target['lab_mono'].shape) == 1:
            _len = 1
            batch_size = target['lab_mono'].shape[0]
        else:
            raise ValueError

        pred = output['out_mono'].view(_len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_mono'].view(-1)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error.item()


class LabCDAccuracy(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        if len(target['lab_cd'].shape) == 2:
            _len = target['lab_cd'].shape[0]
            batch_size = target['lab_cd'].shape[1]

        elif len(target['lab_cd'].shape) == 1:
            _len = 1
            batch_size = target['lab_cd'].shape[0]
        else:
            raise ValueError

        pred = output['out_cd'].view(_len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_cd'].view(-1)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy.item()


class LabCDError(Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        if len(target['lab_cd'].shape) == 2:
            _len = target['lab_cd'].shape[0]
            batch_size = target['lab_cd'].shape[1]

        elif len(target['lab_cd'].shape) == 1:
            _len = 1
            batch_size = target['lab_cd'].shape[0]
        else:
            raise ValueError

        pred = output['out_cd'].view(_len * batch_size, -1).max(dim=1)[1]
        lab = target['lab_cd'].view(-1)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error.item()
