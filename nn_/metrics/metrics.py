import torch
from torch.nn import Module


class LabPhnframeAccuracy(Module):

    def __init__(self):
        super().__init__()

    @property
    def cpu_only(self):
        return False

    def forward(self, output, target):
        pred = output['out_phnframe'].max(dim=1)[1].view(-1)
        lab = target['lab_phnframe'].view(-1)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy.item()


class LabPhnframeError(Module):

    def __init__(self):
        super().__init__()

    @property
    def cpu_only(self):
        return False

    def forward(self, output, target):
        pred = output['out_phnframe'].max(dim=1)[1].view(-1)
        lab = target['lab_phnframe'].view(-1)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error.item()


class LabMonoAccuracy(Module):

    def __init__(self):
        super().__init__()

    @property
    def cpu_only(self):
        return False

    def forward(self, output, target):
        pred = output['out_mono'].max(dim=1)[1].view(-1)
        lab = target['lab_mono'].view(-1)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy.item()


class LabMonoError(Module):

    def __init__(self):
        super().__init__()

    @property
    def cpu_only(self):
        return False

    def forward(self, output, target):
        pred = output['out_mono'].max(dim=1)[1].view(-1)
        lab = target['lab_mono'].view(-1)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error.item()


class LabCDAccuracy(Module):

    def __init__(self):
        super().__init__()

    @property
    def cpu_only(self):
        return False

    def forward(self, output, target):
        if len(output['out_cd'].shape) == 3 and output['out_cd'].shape[1] == 1:
            output['out_cd'] = output['out_cd'].squeeze(1)
        pred = output['out_cd'].max(dim=1)[1].view(-1)
        lab = target['lab_cd'].view(-1).to(dtype=torch.long)

        accuracy = torch.mean((pred == lab).to(dtype=torch.float32))

        return accuracy.item()


class LabCDError(Module):

    def __init__(self):
        super().__init__()

    @property
    def cpu_only(self):
        return False

    def forward(self, output, target):
        if len(output['out_cd'].shape) == 3 and output['out_cd'].shape[1] == 1:
            output['out_cd'] = output['out_cd'].squeeze(1)
        pred = output['out_cd'].max(dim=1)[1].view(-1)
        lab = target['lab_cd'].view(-1).to(dtype=torch.long)

        error = torch.mean((pred != lab).to(dtype=torch.float32))

        return error.item()
