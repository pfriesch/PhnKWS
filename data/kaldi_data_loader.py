import os

import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler

from base.base_kaldi_dataset import BaseKaldiDataset
from data.datasets import DatasetType

from data import PADDING_IGNORE_INDEX


def collate_fn_simple(sample_list):
    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    fea_dict = {k: torch.stack([s[1][k] for s in sample_list]).squeeze(1) for k in fea_keys}
    lab_dict = {k: torch.stack([s[2][k] for s in sample_list]).squeeze(1).to(dtype=torch.int64) for k in lab_keys}
    sample_names = [s[0] for s in sample_list]
    return sample_names, fea_dict, lab_dict


def collate_fn_pad_batch_first(sample_list, feat_padding='zero', ctc_labels=False):
    # TODO compare padding methods
    # feat repeat padding see: https://github.com/SeanNaren/deepspeech.pytorch/issues/312
    # mostly used when batchnorm is used in the model
    # BCT

    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    batch_size = len(sample_list)
    max_length = 0
    for sample in sample_list:
        assert sample[1][fea_keys[0]].shape[-1] == 1
        _len = sample[1][fea_keys[0]].shape[0]
        if _len > max_length:
            max_length = _len

    sample_names = []

    fea_dict = {k: torch.zeros((batch_size, sample_list[0][1][k].shape[1], max_length)) for k in
                fea_keys}
    if not ctc_labels:
        lab_dict = {k: torch.full((batch_size, max_length), fill_value=PADDING_IGNORE_INDEX, dtype=torch.int32) for k in
                    lab_keys}
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

    # .view(-1).to(dtype=torch.int32)

    if ctc_labels:
        assert 'sample_length' not in lab_dict
        assert 'sample_length' not in fea_dict
        lab_dict['target_sequence_lengths'] = target_length
        lab_dict['input_sequence_lengths'] = input_length
    return sample_names, fea_dict, lab_dict


def collate_fn_pad(sample_list, feat_padding='repeat'):
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


class ChunkedStatefulRandomSampler(Sampler):
    r"""Samples elements randomly. State can be saved.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # TODO save sate

    def __init__(self, data_source):
        assert hasattr(data_source, "samples_per_chunk")
        self.data_source = data_source
        samples_per_chunk = data_source.samples_per_chunk

        n = len(self.data_source)
        assert sum(samples_per_chunk) == n
        self.permutation = []
        total_idx = 0
        for chunk_id, chunk_len in enumerate(samples_per_chunk):
            for i in torch.randperm(chunk_len).tolist():
                self.permutation.append(total_idx + i)
            total_idx += chunk_len

        self.start_idx = 0

    def __iter__(self):
        for index in self.permutation[self.start_idx:]:
            # self.start_idx += 1
            yield index

    def __len__(self):
        return len(self.data_source)


class KaldiDataLoader(DataLoader):

    def __init__(self, dataset: BaseKaldiDataset, batch_size, use_gpu, batch_ordering, shuffle=False):
        """

        :param batch_ordering:
        T: sequence length
        B: batch size
        C: channels
        L: appended context length
        """

        assert batch_ordering in ["BCL", "TBCL", "BCT"]

        self.dataset = dataset
        self.n_samples = len(self.dataset)

        # Warn: packed sequence does not work with pin_memory
        pin_memory = use_gpu

        if dataset.state.dataset_type == DatasetType.FRAMEWISE_SHUFFLED_FRAMES:
            assert batch_ordering == "BCL"
            _collate_fn = collate_fn_simple
        else:
            if batch_ordering == "TBCL":
                _collate_fn = collate_fn_pad
            elif batch_ordering == "BCT":
                _collate_fn = collate_fn_pad_batch_first

            else:
                raise ValueError

        if shuffle:
            self._sampler = ChunkedStatefulRandomSampler(dataset)
        else:
            self._sampler = SequentialSampler(dataset)

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              sampler=self._sampler,
                                              collate_fn=_collate_fn,
                                              pin_memory=pin_memory,
                                              num_workers=os.cpu_count() * 2,
                                              drop_last=True)  # drop last because maybe batchnorm
