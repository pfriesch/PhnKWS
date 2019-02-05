import itertools
import random
import copy

import torch
from torch.utils.data import DataLoader, Sampler, RandomSampler

from data import PADDING_IGNORE_INDEX
from data.data_util import chunk_scp
from data.dataset_registry import get_dataset
from data.kaldi_dataset_framewise import KaldiDatasetFramewise
from data.kaldi_dataset_framewise_shuffled_frames import KaldiDatasetFramewiseShuffledFrames
from data.kaldi_dataset_unaligned import KaldiDatasetUnaligned


def collate_fn_simple(sample_list):
    fea_keys = list(sample_list[0][0].keys())
    lab_keys = list(sample_list[0][1].keys())

    fea_dict = {k: torch.stack([s[0][k] for s in sample_list]) for k in fea_keys}
    lab_dict = {k: torch.stack([s[1][k] for s in sample_list]) for k in lab_keys}

    return [], fea_dict, lab_dict


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


class BucketRandomSampler(Sampler):
    def __init__(self, ordering_length, sort_by_feat, n_buckets=None, bucket_size_samples=None):
        assert n_buckets is not bucket_size_samples
        self.ordering_length = ordering_length
        self.feat = sort_by_feat

        ordering_length_items = list(ordering_length[self.feat].items())
        if n_buckets is not None:

            _bucket_length = len(ordering_length_items) // n_buckets

            self.ordering_length_buckets = \
                [ordering_length_items[_bucked_id * _bucket_length:
                                       _bucked_id * _bucket_length + _bucket_length]
                 for _bucked_id in range(n_buckets - 1)]
            self.ordering_length_buckets.append(ordering_length_items[n_buckets - 1 * _bucket_length:])

        elif bucket_size_samples is not None:

            _bucket_length = bucket_size_samples

            _n_buckets = len(ordering_length_items) // _bucket_length

            self.ordering_length_buckets = \
                [ordering_length_items[_bucked_id * _bucket_length:
                                       _bucked_id * _bucket_length + _bucket_length]
                 for _bucked_id in range(_n_buckets - 1)]
            self.ordering_length_buckets.append(ordering_length_items[_n_buckets - 1 * _bucket_length:])

    def __iter__(self):
        for bucket in self.ordering_length_buckets:
            random.shuffle(bucket)
        return iter([idx_length['idx'] for filename, idx_length in
                     itertools.chain(*self.ordering_length_buckets)
                     ])

    def __len__(self):
        return len(self.ordering_length[self.feat])


class SortedSampler(Sampler):
    def __init__(self, ordering_length, sort_by_feat):
        self.ordering_length = ordering_length
        self.feat = sort_by_feat

    def __iter__(self):
        return iter([_file_metadata['idx'] for filename, _file_metadata in self.ordering_length[self.feat].items()])

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
            _collate_fn = collate_fn_zero_pad
            _sampler = BucketRandomSampler(
                self.dataset.ordering_length,
                sort_by_feat=sort_by_feat,
                bucket_size_samples=100) if sort_by_feat is not None \
                else None
        elif isinstance(dataset, KaldiDatasetFramewise):
            _collate_fn = collate_fn_zero_pad
            # SortedSampler(
            #     self.dataset.ordering_length,
            #     sort_by_feat=sort_by_feat) if sort_by_feat is not None
            _sampler = BucketRandomSampler(
                self.dataset.ordering_length,
                sort_by_feat=sort_by_feat,
                bucket_size_samples=100) if sort_by_feat is not None \
                else None
        elif isinstance(dataset, KaldiDatasetFramewiseShuffledFrames):
            _collate_fn = collate_fn_simple
            _sampler = RandomSampler(dataset)
        else:
            raise ValueError

        assert (prefetch_to_gpu and num_workers == 0) or not prefetch_to_gpu

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              sampler=_sampler,
                                              collate_fn=_collate_fn,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              drop_last=True)  # drop last because maybe batchnorm


class KaldiChunkedDataLoader:

    def __init__(self, feature_dict, label_dict, phn_mapping, out_dir,
                 context_left, context_right,
                 max_sequence_length,
                 framewise_labels,
                 tensorboard_logger,

                 batch_size, use_gpu, prefetch_to_gpu,
                 device, sort_by_feat=None):

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
        assert len(set([feature_dict[feature_name]['n_samples_per_chunk'] for feature_name in feature_dict])) == 1

        for feature_name in feature_dict:
            feature_lst_path = feature_dict[feature_name]['feature_lst_path']
            self.n_samples_per_chunk = feature_dict[feature_name]['n_samples_per_chunk']
            self.chunk_paths[feature_name], self.n_samples_per_feat[feature_name] = chunk_scp(feature_lst_path,
                                                                                              self.n_samples_per_chunk,
                                                                                              out_dir)
            self.n_chunks = len(self.chunk_paths[feature_name])

        self.n_samples = sum(self.n_samples_per_feat.values())
        # self.n_batches = sum([n_values // batch_size for n_values in self.n_samples_per_feat.values()])

        self.iterated_over = False

    def __len__(self):
        return -1

    def __iter__(self):
        assert not self.iterated_over, "aussume to generate loader every run"
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
                                      self.tensorboard_logger)

                data_loader = KaldiDataLoader(dataset,
                                              self.batch_size,
                                              self.use_gpu,
                                              self.prefetch_to_gpu,
                                              self.device,
                                              0,
                                              self.sort_by_feat)
                for x in iter(data_loader):
                    yield x
        self.iterated_over = True
