import random

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader, Sampler

from data_loader.kaldi_dataset import KaldiDataset


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

    fea_dict = {k: torch.zeros(max_length, batch_size, sample_list[0][1][k].shape[1]) for k in fea_keys}
    lab_dict = {k: torch.zeros(max_length, batch_size, dtype=torch.int64) for k in lab_keys}
    for _idx, sample in enumerate(sample_list):
        _len = sample[1][fea_keys[0]].shape[0]

        padding_zeros = max_length - _len
        padding_zeros_left = random.randint(0, padding_zeros)

        sample_names.append(sample[0])
        for fea in fea_dict:
            fea_dict[fea][padding_zeros_left: padding_zeros_left + _len, _idx, :] = sample[1][fea]
        for lab in lab_dict:
            lab_dict[lab][padding_zeros_left: padding_zeros_left + _len, _idx] = sample[2][lab]

    return sample_names, fea_dict, lab_dict


def collate_fn(sample_list):
    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    sample_names = []
    fea_dict = {k: list() for k in fea_keys}
    lab_dict = {k: list() for k in lab_keys}
    for sample in sample_list:
        sample_names.append(sample[0])
        for fea in fea_dict:
            fea_dict[fea].append(sample[1][fea])
        for lab in lab_dict:
            lab_dict[lab].append(sample[2][lab])

    for fea in fea_dict:
        fea_dict[fea] = pack_sequence(sorted(fea_dict[fea], key=lambda x: x.shape[0], reverse=True))

    for lab in lab_dict:
        lab_dict[lab] = pack_sequence(sorted(lab_dict[lab], key=lambda x: x.shape[0], reverse=True))

    return sample_names, fea_dict, lab_dict


class SortedSampler(Sampler):
    def __init__(self, ordering_length, sort_by_feat):
        self.ordering_length = ordering_length
        self.feat = sort_by_feat

    def __iter__(self):
        return iter([idx_length['idx'] for filename, idx_length in self.ordering_length[self.feat].items()])

    def __len__(self):
        return len(self.ordering_length[self.feat])


class KaldiDataLoader(DataLoader):

    def __init__(self, dataset: KaldiDataset, batch_size, use_gpu, prefetch_to_gpu,
                 device, num_workers, sort_by_feat=None):
        self.dataset = dataset
        self.n_samples = len(self.dataset)
        if prefetch_to_gpu:
            self.dataset.move_to(device)

        # pin_memory = use_gpu and not prefetch_to_gpu
        # TODO packed sequence from collate_fn does not work with pin_memory
        pin_memory = False

        assert (prefetch_to_gpu and num_workers == 0) or not prefetch_to_gpu

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              sampler=
                                              SortedSampler(
                                                  self.dataset.ordering_length,
                                                  sort_by_feat=sort_by_feat) if sort_by_feat is not None
                                              else None,
                                              collate_fn=collate_fn_zero_pad,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              drop_last=False)
