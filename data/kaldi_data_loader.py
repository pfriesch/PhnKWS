import os
from functools import partial

import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler

from base.base_kaldi_dataset import BaseKaldiDataset
from data.datasets import DatasetType

from data import PADDING_IGNORE_INDEX
from utils.logger_config import logger


def collate_fn_simple(sample_list):
    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    fea_dict = {k: torch.stack([s[1][k] for s in sample_list]).squeeze(1) for k in fea_keys}
    lab_dict = {k: torch.stack([s[2][k] for s in sample_list]).to(dtype=torch.int64) for k in lab_keys}
    sample_names = [s[0] for s in sample_list]
    return sample_names, fea_dict, lab_dict


def collate_fn_pad_batch_first(sample_list, feat_padding='zero', ctc_labels=False):
    # TODO compare padding methods
    # feat repeat padding see: https://github.com/SeanNaren/deepspeech.pytorch/issues/312
    # mostly used when batchnorm is used in the model
    # NCL

    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    batch_size = len(sample_list)
    max_length_feat = 0
    for sample in sample_list:
        assert sample[1][fea_keys[0]].shape[-1] == 1
        _len = sample[1][fea_keys[0]].shape[0]
        if _len > max_length_feat:
            max_length_feat = _len

    if not ctc_labels:

        batch_size = len(sample_list)
        max_length_labs = 0
        for sample in sample_list:
            _len = sample[2][lab_keys[0]].shape[0]
            if _len > max_length_labs:
                max_length_labs = _len

    sample_names = []

    fea_dict = {k: torch.zeros((batch_size, sample_list[0][1][k].shape[1], max_length_feat)) for k in
                fea_keys}
    if not ctc_labels:
        lab_dict = {k: torch.full((batch_size, max_length_labs),
                                  fill_value=PADDING_IGNORE_INDEX,
                                  dtype=torch.int64) for k in lab_keys}
    else:
        lab_dict = {k: [[]] * batch_size for k in
                    lab_keys}
    input_length = torch.zeros(batch_size, dtype=torch.int64)
    target_length = torch.zeros(batch_size, dtype=torch.int64)

    for _idx, sample in enumerate(sample_list):
        _len_feat = sample[1][fea_keys[0]].shape[0]
        _len_lab = sample[2][lab_keys[0]].shape[0]

        sample_names.append(sample[0])
        for fea in fea_dict:
            fea_dict[fea][_idx, :, :_len_feat] = sample[1][fea].squeeze(2).t()
            if feat_padding == 'repeat':
                raise NotImplementedError
                # padding_left = len(fea_dict[fea]) - _len_feat
                # if padding_left > 0:
                #     total_padded = 0
                #     while (total_padded < padding_left):
                #         # if fea_dict[fea][_len_feat:, _idx, :, :].shape != sample[1][fea][:padding_left].shape:
                #         #         print(fea_dict[fea][_len_feat:, _idx, :, :].shape, sample[1][fea][:padding_left].shape)
                #         if len(fea_dict[fea][_len_feat:, _idx, :, :]) > len(sample[1][fea]):
                #             fea_dict[fea][
                #             _len_feat + total_padded:_len_feat + total_padded + len(sample[1][fea]),
                #             _idx, :, :] = \
                #                 sample[1][fea][:padding_left - total_padded]
                #             total_padded += len(sample[1][fea])
                #
                #         else:
                #             fea_dict[fea][_len_feat:, _idx, :, :] = sample[1][fea][:padding_left]
                #             total_padded += len(sample[1][fea][:padding_left])

        for lab in lab_keys:
            if not ctc_labels:
                lab_dict[lab][_idx][:_len_lab] = sample[2][lab]
            else:
                lab_dict[lab][_idx] = sample[2][lab]

        input_length[_idx] = _len_feat
        target_length[_idx] = _len_lab

    sorting = sorted(range(len(input_length)), key=lambda k: input_length[k], reverse=True)
    input_length = input_length[sorting]
    target_length = target_length[sorting]

    fea_dict = {k: fea_dict[k][sorting] for k in fea_keys}
    if not ctc_labels:
        lab_dict = {k: lab_dict[k][sorting] for k in lab_dict}

    else:
        lab_dict = {k: torch.cat([lab_dict[k][i] for i in sorting], dim=0) for k in lab_keys}
        for k in lab_dict:
            assert 0 not in lab_dict[k]

    # .view(-1).to(dtype=torch.int32)

    if ctc_labels:
        assert 'sample_length' not in lab_dict
        assert 'sample_length' not in fea_dict
        lab_dict['target_sequence_lengths'] = target_length
        lab_dict['input_sequence_lengths'] = input_length
    return sample_names, fea_dict, lab_dict


def collate_fn_pad(sample_list, feat_padding='repeat', ctc_labels=True):
    # TODO compare padding methods
    # feat repeat padding see: https://github.com/SeanNaren/deepspeech.pytorch/issues/312
    # mostly used when batchnorm is used in the model

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
    lab_dict = {k: [[]] * batch_size for k in
                lab_keys}
    input_length = torch.zeros(batch_size, dtype=torch.int64)
    target_length = torch.zeros(batch_size, dtype=torch.int64)

    for _idx, sample in enumerate(sample_list):
        _len_feat = sample[1][fea_keys[0]].shape[0]
        _len_lab = sample[2][lab_keys[0]].shape[0]

        sample_names.append(sample[0])
        for fea in fea_dict:
            fea_dict[fea][:_len_feat, _idx, :, :] = sample[1][fea]
            if feat_padding == 'repeat':
                padding_left = len(fea_dict[fea]) - _len_feat
                if padding_left > 0:
                    total_padded = 0
                    while (total_padded < padding_left):
                        # if fea_dict[fea][_len_feat:, _idx, :, :].shape != sample[1][fea][:padding_left].shape:
                        #         print(fea_dict[fea][_len_feat:, _idx, :, :].shape, sample[1][fea][:padding_left].shape)
                        if len(fea_dict[fea][_len_feat:, _idx, :, :]) > len(sample[1][fea]):
                            fea_dict[fea][
                            _len_feat + total_padded:_len_feat + total_padded + len(sample[1][fea]),
                            _idx, :, :] = \
                                sample[1][fea][:padding_left - total_padded]
                            total_padded += len(sample[1][fea])

                        else:
                            fea_dict[fea][_len_feat:, _idx, :, :] = sample[1][fea][:padding_left]
                            total_padded += len(sample[1][fea][:padding_left])

        for lab in lab_keys:
            lab_dict[lab][_idx] = sample[2][lab]

        input_length[_idx] = _len_feat
        target_length[_idx] = _len_lab

    sorting = sorted(range(len(input_length)), key=lambda k: input_length[k], reverse=True)
    input_length = input_length[sorting]
    target_length = target_length[sorting]

    fea_dict = {k: fea_dict[k][:, sorting] for k in fea_keys}
    lab_dict = {k: torch.cat([lab_dict[k][i] for i in sorting], dim=0) for k in lab_keys}

    # .view(-1).to(dtype=torch.int32)

    assert 'sample_length' not in lab_dict
    assert 'sample_length' not in fea_dict
    lab_dict['target_sequence_lengths'] = target_length
    lab_dict['input_sequence_lengths'] = input_length
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


class StatefulChunkedRandomSampler(Sampler):
    r"""Samples elements randomly. State can be saved.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # TODO save sate

    def __init__(self, data_source):
        assert hasattr(data_source, "samples_per_chunk")
        self.data_source = data_source
        self.samples_per_chunk = data_source.samples_per_chunk

        n = len(self.data_source)
        assert sum(self.samples_per_chunk) == n
        self.permutation = []
        total_idx = 0
        for chunk_id, chunk_len in enumerate(self.samples_per_chunk):
            for i in torch.randperm(chunk_len).tolist():
                self.permutation.append(total_idx + i)
            total_idx += chunk_len

        self.start_idx = 0

    def __iter__(self):
        for self.start_idx, index in enumerate(self.permutation[self.start_idx:], start=self.start_idx):
            # self.start_idx += 1
            yield index

    def __len__(self):
        return len(self.data_source)

    def state_dict(self):
        return {'permutation': self.permutation,
                'start_idx': self.start_idx,
                'samples_per_chunk': self.samples_per_chunk}

    def load_state_dict(self, state_dict):
        if self.samples_per_chunk == state_dict['samples_per_chunk']:
            self.permutation = state_dict['permutation']
            self.start_idx = state_dict['start_idx'] + 1
        else:
            logger.warn("The dataset used when this sampler was saved is not the same as the one used now.\n"
                        "Ignoring the saved sampler and restarting sampling.")


class StatefulSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.start_idx = 0

    def __iter__(self):
        for self.start_idx, index in \
                list(enumerate(range(len(self.data_source)), start=self.start_idx))[self.start_idx:]:
            yield index


    def __len__(self):
        return len(self.data_source)

    def state_dict(self):
        return {
            'start_idx': self.start_idx,
            'data_source_len': len(self.data_source)}

    def load_state_dict(self, state_dict):
        if len(self.data_source) == state_dict['data_source_len']:
            self.start_idx = state_dict['start_idx'] + 1
        else:
            logger.warn("The dataset used when this sampler was saved is not the same as the one used now.\n"
                        "Ignoring the saved sampler and restarting sampling.")


class KaldiDataLoader(DataLoader):

    def __init__(self, dataset: BaseKaldiDataset, batch_size, use_gpu, batch_ordering, shuffle=False):
        """

        :param batch_ordering:
        T: sequence length
        B: batch size
        C: channels
        L: appended context length
        """

        assert batch_ordering in ["NCL", "TNCL", "NCL"]

        self.dataset = dataset
        self.n_samples = len(self.dataset)

        # Warn: packed sequence does not work with pin_memory
        pin_memory = use_gpu

        # FRAMEWISE_SEQUENTIAL = 2
        # FRAMEWISE_SEQUENTIAL_APPENDED_CONTEXT = 3
        # SEQUENTIAL = 4
        # SEQUENTIAL_APPENDED_CONTEXT = 5

        if dataset.state.dataset_type == DatasetType.FRAMEWISE_SHUFFLED_FRAMES:
            assert batch_ordering == "NCL"
            _collate_fn = collate_fn_simple
        else:
            if batch_ordering == "TNCL":
                if dataset.state.dataset_type == DatasetType.SEQUENTIAL_APPENDED_CONTEXT:
                    _collate_fn = partial(collate_fn_pad, feat_padding='zero', ctc_labels=True)
                elif dataset.state.dataset_type == DatasetType.FRAMEWISE_SEQUENTIAL_APPENDED_CONTEXT:
                    _collate_fn = partial(collate_fn_pad, feat_padding='zero', ctc_labels=False)
                else:
                    raise NotImplementedError
                # _collate_fn = collate_fn_pad
            elif batch_ordering == "NCL":
                if dataset.state.dataset_type == DatasetType.FRAMEWISE_SEQUENTIAL:
                    _collate_fn = partial(collate_fn_pad_batch_first, feat_padding='zero', ctc_labels=False)
                elif dataset.state.dataset_type == DatasetType.SEQUENTIAL:
                    _collate_fn = partial(collate_fn_pad_batch_first, feat_padding='zero', ctc_labels=True)

                elif dataset.state.dataset_type == DatasetType.SEQUENTIAL_APPENDED_CONTEXT:
                    _collate_fn = partial(collate_fn_pad, feat_padding='zero', ctc_labels=True)
                elif dataset.state.dataset_type == DatasetType.FRAMEWISE_SEQUENTIAL_APPENDED_CONTEXT:
                    _collate_fn = partial(collate_fn_pad, feat_padding='zero', ctc_labels=False)
                else:
                    raise ValueError
            else:
                raise ValueError

        if shuffle:
            _sampler = StatefulChunkedRandomSampler(dataset)
        else:
            _sampler = StatefulSequentialSampler(dataset)

        if 'DEBUG_MODE' in os.environ and os.environ['DEBUG_MODE']:
            _num_workers = 0
        else:
            _num_workers = os.cpu_count() * 2

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              sampler=_sampler,
                                              collate_fn=_collate_fn,
                                              pin_memory=pin_memory,
                                              num_workers=_num_workers,
                                              drop_last=True)  # drop last because maybe batchnorm

    def start(self):
        if self.drop_last:
            return self.sampler.start_idx // self.batch_size
        else:
            return (self.sampler.start_idx + self.batch_size - 1) // self.batch_size
