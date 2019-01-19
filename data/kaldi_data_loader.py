import random
import copy

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader, Sampler
import numpy as np

from data.data_util import chunk_scp
from data.dataset_registry import get_dataset
from data.kaldi_dataset_framewise import KaldiDatasetFramewise
from data.kaldi_dataset_unaligned import KaldiDatasetUnaligned


def collate_fn_rnd_zero_pad(sample_list):
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
        lab_dict[lab] = sorted(lab_dict[lab], key=lambda x: x.shape[0], reverse=True)
        sequence_length = torch.from_numpy(np.array([len(l) for l in lab_dict[lab]]))
        lab_dict[lab] = {"label": pack_sequence(lab_dict[lab]), "sequence_lengths": sequence_length}

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

    def __init__(self, dataset, batch_size, use_gpu, prefetch_to_gpu,
                 device, num_workers, sort_by_feat=None):
        self.dataset = dataset
        self.n_samples = len(self.dataset)
        if prefetch_to_gpu:
            self.dataset.move_to(device)
            assert num_workers == 0

        # pin_memory = use_gpu and not prefetch_to_gpu
        # TODO packed sequence from collate_fn does not work with pin_memory but we prefetch to gpu anyway
        pin_memory = False

        if isinstance(dataset, KaldiDatasetUnaligned):
            _collate_fn = collate_fn
        elif isinstance(dataset, KaldiDatasetFramewise):
            _collate_fn = collate_fn_rnd_zero_pad
        else:
            raise ValueError

        assert (prefetch_to_gpu and num_workers == 0) or not prefetch_to_gpu

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              sampler=
                                              SortedSampler(
                                                  self.dataset.ordering_length,
                                                  sort_by_feat=sort_by_feat) if sort_by_feat is not None
                                              else None,
                                              collate_fn=_collate_fn,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              drop_last=False)


class KaldiChunkedDataLoader:

    def __init__(self, feature_dict, label_dict, phn_mapping, out_dir,
                 context_left, context_right,
                 max_sequence_length,
                 framewise_labels,
                 tensorboard_logger,

                 batch_size, use_gpu, prefetch_to_gpu,
                 device, sort_by_feat=None, debug=False, local=False):

        #### DATASET
        self.feature_dict = feature_dict
        self.label_dict = label_dict
        self.phn_mapping = phn_mapping
        self.out_dir = out_dir
        self.context_left = context_left
        self.context_right = context_right
        self.max_sequence_length = max_sequence_length
        self.framewise_labels = framewise_labels
        self.tensorboard_logger = tensorboard_logger
        self.debug = debug
        self.local = local

        #### DATALOADER

        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.prefetch_to_gpu = prefetch_to_gpu
        self.device = device
        self.sort_by_feat = sort_by_feat

        #### Chunking

        self.chunk_paths = {}
        self.n_samples_per_feat = {}

        # all featers have same amount of chunks
        assert len(set([feature_dict[feature_name]['n_chunks'] for feature_name in feature_dict])) == 1

        for feature_name in feature_dict:
            feature_lst_path = feature_dict[feature_name]['feature_lst_path']
            self.n_chunks = feature_dict[feature_name]['n_chunks']
            self.chunk_paths[feature_name], self.n_samples_per_feat[feature_name] = chunk_scp(feature_lst_path,
                                                                                              self.n_chunks,
                                                                                              out_dir)

        self.n_samples = sum(self.n_samples_per_feat.values())

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        for _chunk_id in range(self.n_chunks):
            for feature_name in self.chunk_paths:
                _feature_dict = copy.deepcopy(self.feature_dict)

                _feature_dict[feature_name]['feature_lst_path'] = self.chunk_paths[feature_name][_chunk_id]

                dataset = get_dataset(_feature_dict,
                                      self.label_dict,
                                      self.phn_mapping,
                                      self.context_left,
                                      self.context_right,
                                      self.max_sequence_length,
                                      self.framewise_labels,
                                      self.tensorboard_logger,
                                      self.debug,
                                      self.local)

                data_loader = KaldiDataLoader(dataset,
                                              self.batch_size,
                                              self.use_gpu,
                                              self.prefetch_to_gpu,
                                              self.device,
                                              0,
                                              self.sort_by_feat)
                for x in iter(data_loader):
                    yield x
