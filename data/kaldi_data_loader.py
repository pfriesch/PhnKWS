import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

from data import PADDING_IGNORE_INDEX
from data.kaldi_dataset import KaldiDataset


def collate_fn_simple(sample_list):
    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    fea_dict = {k: torch.stack([s[1][k] for s in sample_list]).squeeze(1) for k in fea_keys}
    lab_dict = {k: torch.stack([s[2][k] for s in sample_list]).squeeze(1).to(dtype=torch.int64) for k in lab_keys}
    sample_names = [s[0] for s in sample_list]
    return sample_names, fea_dict, lab_dict


def collate_fn_zero_pad(sample_list):
    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    batch_size = len(sample_list)
    max_length = 0
    for sample in sample_list:
        _len = sample[1][fea_keys[0]].shape[0]
        if _len > max_length:
            max_length = _len

    sample_names = []

    fea_dict = {k: torch.zeros([max_length, batch_size] + list(sample_list[0][1][k].shape[1:])) for k in fea_keys}
    lab_dict = {k: torch.full((max_length, batch_size), dtype=torch.int64, fill_value=-PADDING_IGNORE_INDEX) for k in
                lab_keys}
    for _idx, sample in enumerate(sample_list):
        _len_feat = sample[1][fea_keys[0]].shape[0]
        _len_lab = sample[2][lab_keys[0]].shape[0]

        sample_names.append(sample[0])
        for fea in fea_dict:
            fea_dict[fea][:_len_feat, _idx, :, :] = sample[1][fea]
        for lab in lab_dict:
            lab_dict[lab][:_len_lab, _idx] = sample[2][lab]

    return sample_names, fea_dict, lab_dict


class StatefulRandomSampler(Sampler):
    r"""Samples elements randomly. State can be saved.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # TODO save sate

    def __init__(self, data_source):
        self.data_source = data_source

        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        self.start_idx = 0

    def __iter__(self):
        for index in self.permutation[self.start_idx:]:
            # self.start_idx += 1
            yield index

    def __len__(self):
        return len(self.data_source)


class KaldiDataLoader(DataLoader):

    def __init__(self, dataset: KaldiDataset, batch_size, use_gpu,
                 num_workers):
        self.dataset = dataset
        self.n_samples = len(self.dataset)

        # Warn: packed sequence does not work with pin_memory
        pin_memory = use_gpu

        if dataset.shuffle_frames:
            _collate_fn = collate_fn_simple
        else:
            _collate_fn = collate_fn_zero_pad
        self._sampler = StatefulRandomSampler(dataset)

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              sampler=self._sampler,
                                              collate_fn=_collate_fn,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              drop_last=True)  # drop last because maybe batchnorm

    def set_sampler_start_idx(self, index):
        self._sampler.curr_idx = index
