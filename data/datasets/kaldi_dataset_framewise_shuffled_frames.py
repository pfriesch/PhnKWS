import os
from collections import Counter

import torch
import numpy as np

from base.base_kaldi_dataset import BaseKaldiDataset
from data.data_util import load_labels
from base.base_kaldi_dataset import DatasetType


class KaldiDatasetFramewiseContextShuffledFrames(BaseKaldiDataset):

    def __init__(self, data_cache_root, dataset_name, feature_dict, label_dict,
                 left_context=0, right_context=0, normalize_features=True,
                 overfit_small_batch=False):

        dataset_type = DatasetType.FRAMEWISE_SHUFFLED_FRAMES

        for label_name in label_dict:
            assert label_dict[label_name]["label_opts"] == "ali-to-phones --per-frame=true" or \
                   label_dict[label_name]["label_opts"] == "ali-to-pdf"

        super().__init__(data_cache_root, dataset_name, feature_dict, label_dict, dataset_type, max_sample_len=None,
                         left_context=left_context, right_context=right_context, normalize_features=normalize_features,
                         max_seq_len=None, max_label_length=None,
                         overfit_small_batch=overfit_small_batch)
        self.state.aligned_labels = True

    @property
    def shuffle_frames(self):
        return True

    @property
    def samples_per_chunk(self):
        assert 'sample_index' in self.sample_index
        return list(Counter([s[0] for s in self.sample_index['sample_index']]).values())

    def __len__(self):
        assert 'sample_index' in self.sample_index
        return len(self.sample_index['sample_index'])

    def _getitem(self, index):
        # # Reason for not just doing a big matrix with all samples and skipping the following mess:
        # # * Since we want to save disk space it is not feasible to precompute the context windows and append
        # #   them to the samples and save this. Therefore we have to add the context here while loading.
        # # * We also want to know the total number of frames in the dataset to return a length.
        #

        # _sample_index = self.sample_index[index]
        chunk_idx, file_idx, in_sample_index = self.sample_index['sample_index'][index]
        filename = self.sample_index['filenames'][file_idx]

        if self.cached_pt != chunk_idx:
            self.cached_pt = chunk_idx
            self.cached_samples = torch.load(
                os.path.join(self.state.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

        sample = self.cached_samples['samples'][filename]
        lables = {}
        for label_name in sample['labels']:
            lables[label_name] = np.expand_dims(sample['labels'][label_name][in_sample_index], 0)

        features = {}
        for feature_name in sample['features']:
            features[feature_name] = \
                np.expand_dims(sample['features'][feature_name][
                               in_sample_index:
                               in_sample_index + self.state.left_context + self.state.right_context + 1, :].T, 0)

        return filename, features, lables

    def _load_labels(self, label_dict):

        all_labels_loaded = {}

        for label_name in label_dict:
            all_labels_loaded[label_name] = load_labels(label_dict[label_name]['label_folder'],
                                                        label_dict[label_name]['label_opts'])
            assert self.state.max_label_length is None

            assert label_name != "lab_phn" \
                   and label_name == "lab_cd" or label_name == "lab_mono"

            self._check_labels_indexed_from(all_labels_loaded, label_name)

        return all_labels_loaded

    def _filenames_from_sample_index(self):
        """

        :return: filenames, _sample_index
        """
        filenames = {filename: file_idx
                     for file_idx, filename in enumerate(Counter([filename
                                                                  for _, filename, _ in
                                                                  self.sample_index]).keys())}

        _sample_index = np.array([(chunk_idx, filenames[filename], sample_idx)
                                  for chunk_idx, filename, sample_idx in self.sample_index], dtype=np.int32)

        return filenames, _sample_index
