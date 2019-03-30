from __future__ import print_function

import os
import os.path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from base.base_kaldi_dataset import BaseKaldiDataset, apply_context_full_sequence
from data.data_util import load_labels
from data.datasets import DatasetType
from data.phoneme_dict import PhonemeDict


class KaldiDatasetFramewise(BaseKaldiDataset):

    def __init__(self, data_cache_root, dataset_name, feature_dict, label_dict, phoneme_dict, max_sample_len=1000,
                 left_context=0, right_context=0, normalize_features=True, max_seq_len=100, max_label_length=None,
                 overfit_small_batch=False):

        dataset_type = DatasetType.FRAMEWISE_SEQUENTIAL
        for label_name in label_dict:
            assert label_dict[label_name]["label_opts"] == "ali-to-phones --per-frame=true" or \
                   label_dict[label_name]["label_opts"] == "ali-to-pdf"

        assert isinstance(phoneme_dict, PhonemeDict)
        super().__init__(data_cache_root, dataset_name, feature_dict, label_dict, dataset_type, max_sample_len,
                         left_context + right_context + 50,
                         left_context, right_context, normalize_features,
                         aligned_labels=True, max_seq_len=max_seq_len,
                         max_label_length=max_label_length,
                         phoneme_dict=phoneme_dict)
        if overfit_small_batch:
            self.sample_index = self.sample_index[:64]

    @property
    def shuffle_frames(self):
        return False

    @property
    def samples_per_chunk(self):
        return list(Counter([s[0] for s in self.sample_index]).values())

    def __len__(self):
        return len(self.sample_index)

    def _getitem(self, index):
        #     context left    context right
        #           v            v
        #         |---|         |-|
        #          _ _ _ _ _ _ _ _
        #         |   |         | |
        #         |   | frames  | |
        #         |_ _|_ _ _ _ _|_|
        #             ^         ^
        #           start      end
        #            index      index
        #
        chunk_idx, file_idx, start_idx, end_idx = self.sample_index[index]
        filename = self.filename_index[file_idx]

        # assert end_idx - start_idx <= self.state.max_seq_len + self.state.max_seq_len // 4 and end_idx - start_idx >= self.state.max_seq_len // 4, \
        #     f"got {end_idx - start_idx} expeced {end_idx - start_idx} <= {self.state.max_seq_len + self.state.max_seq_len // 4} " \
        #     + f"and {end_idx - start_idx} >= {self.state.max_seq_len // 4}"

        if self.cached_pt != chunk_idx:
            self.cached_pt = chunk_idx
            self.cached_samples = torch.load(
                os.path.join(self.state.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

        features, lables = apply_context_full_sequence(self.cached_samples['samples'][filename],
                                                       start_idx=start_idx, end_idx=end_idx,
                                                       context_right=self.state.right_context,
                                                       context_left=self.state.left_context,
                                                       aligned_labels=self.state.aligned_labels)
        # for feature_name in features:
        #     if not end_idx - start_idx == len(features[feature_name]):
        #         assert end_idx - start_idx == len(features[feature_name]), \
        #             f"{end_idx - start_idx} =!= {len(features[feature_name])}"
        #     assert features[feature_name].shape[-1] == 1  # no context appended

        return filename, features, lables

    def _load_labels(self, label_dict):

        all_labels_loaded = {}

        for label_name in label_dict:
            all_labels_loaded[label_name] = load_labels(label_dict[label_name]['label_folder'],
                                                        label_dict[label_name]['label_opts'])
            if self.state.max_label_length is not None and self.state.max_label_length > 0:
                all_labels_loaded[label_name] = \
                    {l: all_labels_loaded[label_name][l] for l in all_labels_loaded[label_name]
                     if len(all_labels_loaded[label_name][l]) < self.state.max_label_length}

            assert label_name != "lab_phn" \
                   and label_name == "lab_cd" or label_name == "lab_mono" or label_name == "lab_phnframe"

            if label_name == "lab_phnframe":
                assert self.state.phoneme_dict is not None
                for sample_id in tqdm(all_labels_loaded[label_name], desc="mapping phnframe"):
                    assert max(all_labels_loaded[label_name][sample_id]) <= max(
                        self.state.phoneme_dict.idx2reducedIdx.keys()), \
                        "Are you sure you have the righ phoneme dict?" + \
                        " Labels have higher indices than phonemes ( {} <!= {} )".format(
                            max(all_labels_loaded[label_name][sample_id]),
                            max(self.state.phoneme_dict.idx2reducedIdx.keys()))

                    # map labels according to phoneme dict
                    tmp_labels = np.copy(all_labels_loaded[label_name][sample_id])
                    for k, v in self.state.phoneme_dict.idx2reducedIdx.items():
                        tmp_labels[all_labels_loaded[label_name][sample_id] == k] = v

                    all_labels_loaded[label_name][sample_id] = tmp_labels

            self._check_labels_indexed_from(all_labels_loaded, label_name)

        return all_labels_loaded

    def _filenames_from_sample_index(self, sample_index):
        """

        :return: filenames, _sample_index
        """
        filenames = {filename: file_idx
                     for file_idx, filename in enumerate(Counter([filename
                                                                  for _, filename, _, _ in
                                                                  sample_index]).keys())}

        _sample_index = np.array([(chunk_idx, filenames[filename], start_idx, end_idx)
                                  for chunk_idx, filename, start_idx, end_idx in sample_index], dtype=np.int32)
        return filenames, _sample_index
