from __future__ import print_function

import json
import os
import os.path
import random
import errno
import sys
from collections import Counter
from functools import partial
from multiprocessing.pool import Pool
from types import SimpleNamespace

import numpy as np
import torch.utils.data as data
import torch
from tqdm import tqdm

from data.data_util import split_chunks, apply_context_single_feat
from data.kaldi_dataset_utils import convert_chunk_from_kaldi_format, _load_labels
from data.phoneme_dict import load_phoneme_dict, PhonemeDict
from utils.logger_config import logger


# inspired by https://github.com/pytorch/audio/blob/master/torchaudio/datasets/vctk.py
class KaldiDataset(data.Dataset):
    """
    Termenology:
    Chunk: A number of files/samples put together in one file to cache
    Split: A sample that is split up by length using the forced aligned labels

    """
    dataset_prefix = "kaldi"
    info_filename = "info.json"
    sample_index_filename = "sample_index.npz"

    def __init__(self, data_cache_root,
                 dataset_name,
                 feature_dict,
                 label_dict,
                 device,
                 max_sample_len=1000,
                 left_context=0,
                 right_context=0,
                 normalize_features=True,
                 phoneme_dict=None,  # e.g. kaldi/egs/librispeech/s5/data/lang/phones.txt

                 max_seq_len=100,
                 max_label_length=None,
                 shuffle_frames=False,
                 overfit_small_batch=False
                 ):
        self.state = SimpleNamespace()

        self.state.overfit_small_batch = overfit_small_batch
        assert isinstance(phoneme_dict, PhonemeDict)

        self.state.aligned_labels = {}

        for label_name in label_dict:
            if label_dict[label_name]["label_opts"] == "ali-to-phones --per-frame=true" or \
                    label_dict[label_name]["label_opts"] == "ali-to-pdf":
                self.state.aligned_labels[label_name] = True

            elif label_dict[label_name]["label_opts"] == "ali-to-phones":
                self.state.aligned_labels[label_name] = False
                if max_seq_len < 0:
                    max_seq_len = None

            else:
                raise NotImplementedError

        self.state.shuffle_frames = shuffle_frames
        if self.state.shuffle_frames:
            assert max_label_length is None
            assert max_seq_len is False or \
                   max_seq_len is None
            assert max_sample_len is False or \
                   max_sample_len is None
        if max_sample_len is None or max_sample_len < 0:
            max_sample_len = float("inf")

        self.state.data_cache_root = os.path.expanduser(data_cache_root)
        self.state.chunk_size = 100  # 1000 for TIMIT & 100 for libri
        self.state.max_len_per_chunk = None
        self.state.min_len_per_chunk = None
        self.state.max_seq_len = max_seq_len
        self.state.max_sample_len = max_sample_len
        self.state.min_sample_len = 0  # TODO ?
        self.state.max_label_length = max_label_length
        self.state.left_context = left_context
        self.state.right_context = right_context
        self.state.phoneme_dict = phoneme_dict
        self.state.label_index_from = 1  # TODO ?

        self.cached_pt = 0
        self.sample_index = []

        self.state.normalize_features = normalize_features

        self.state.dataset_path = os.path.join(self.state.data_cache_root, self.dataset_prefix, "processed",
                                               dataset_name)
        if not self._check_exists():
            self._convert_from_kaldi_format(feature_dict, label_dict)
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        self._read_info()

        self.cached_samples = torch.load(
            os.path.join(self.state.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

    @property
    def shuffle_frames(self):
        return self.state.shuffle_frames

    @property
    def samples_per_chunk(self):
        if 'sample_index' in self.sample_index:
            return list(Counter([s[0] for s in self.sample_index['sample_index']]).values())

        else:
            return list(Counter([s[0] for s in self.sample_index]).values())

    def __len__(self):
        if 'sample_index' in self.sample_index:
            return len(self.sample_index['sample_index'])
        else:
            return len(self.sample_index)

    def _getitem_shuffled_frames(self, index):
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

    def _getitem_sequential(self, index):
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
        chunk_idx, file_idx, start_idx, end_idx = self.sample_index['sample_index'][index]
        filename = self.sample_index['filenames'][file_idx]

        if self.cached_pt != chunk_idx:
            self.cached_pt = chunk_idx
            self.cached_samples = torch.load(
                os.path.join(self.state.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

        features, lables = apply_context(self.cached_samples['samples'][filename],
                                         start_idx=start_idx, end_idx=end_idx,
                                         context_right=self.state.right_context, context_left=self.state.left_context,
                                         aligned_labels=self.state.aligned_labels)
        for feature_name in features:
            if not end_idx - start_idx == len(features[feature_name]):
                assert end_idx - start_idx == len(features[feature_name]), \
                    f"{end_idx - start_idx} =!= {len(features[feature_name])}"

        return filename, features, lables

    def __getitem__(self, index):
        if self.state.shuffle_frames:
            filename, features, lables = self._getitem_shuffled_frames(index)

        else:
            filename, features, lables = self._getitem_sequential(index)

        if self.state.normalize_features:
            # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
            for feature_name in features:
                features[feature_name] = (features[feature_name] -
                                          np.expand_dims(self.cached_samples['means'][feature_name], axis=-1)) / \
                                         np.expand_dims(self.cached_samples['std'][feature_name], axis=-1)

        return filename, np_dict_to_torch(features), np_dict_to_torch(lables)

    def _convert_from_kaldi_format(self, feature_dict, label_dict):
        logger.info("Converting features from kaldi features!")
        main_feat = next(iter(feature_dict))

        # download files
        try:
            os.makedirs(self.state.dataset_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        all_labels_loaded = _load_labels(label_dict, self.state.label_index_from, self.state.max_label_length,
                                         self.state.phoneme_dict)

        with open(feature_dict[main_feat]["feature_lst_path"], "r") as f:
            lines = f.readlines()
        feat_list = lines
        random.shuffle(feat_list)
        file_chunks = list(split_chunks(feat_list, self.state.chunk_size, self.state.overfit_small_batch))

        self.state.max_len_per_chunk = [0] * len(file_chunks)
        self.state.min_len_per_chunk = [sys.maxsize] * len(file_chunks)

        _convert_chunk_from_kaldi_format = partial(convert_chunk_from_kaldi_format,
                                                   dataset_path=self.state.dataset_path,
                                                   feature_dict=feature_dict,
                                                   label_dict=label_dict,
                                                   all_labels_loaded=all_labels_loaded,
                                                   shuffle_frames=self.state.shuffle_frames,
                                                   main_feat=main_feat,
                                                   aligned_labels=self.state.aligned_labels,
                                                   max_sample_len=self.state.max_sample_len,
                                                   min_sample_len=self.state.min_sample_len,
                                                   max_seq_len=self.state.max_seq_len,
                                                   left_context=self.state.left_context,
                                                   right_context=self.state.right_context)
        with tqdm(total=len(file_chunks)) as pbar:
            with Pool() as pool:
                chunksize = len(file_chunks) // (
                        2 * os.cpu_count())
                if chunksize < 1:
                    chunksize = 1
                for chnk_id, sample_splits, max_len, min_len in pool.imap_unordered(_convert_chunk_from_kaldi_format,
                                                                                    enumerate(file_chunks),
                                                                                    chunksize=chunksize):
                    self.sample_index.extend(sample_splits)
                    if max_len is not None:
                        self.state.max_len_per_chunk[chnk_id] = max_len
                    if min_len is not None:
                        self.state.min_len_per_chunk[chnk_id] = min_len
                    pbar.update()

        assert len(self) > 0, \
            f"No sample with a max seq length of {self.state.max_seq_len} in the dataset! " \
            + f"Try to choose a higher max seq length to start with"
        self._write_info()
        logger.info('Done extracting kaldi features!')

    def _check_exists(self):
        if not os.path.exists(os.path.join(self.state.dataset_path, self.info_filename)):
            return False
        else:
            with open(os.path.join(self.state.dataset_path, self.info_filename), "r") as f:
                _state = json.load(f)
                if not set(vars(self.state).keys()) == set(_state.keys()):
                    return False
                for k, v in vars(self.state).items():
                    if k not in ['max_len_per_chunk', 'min_len_per_chunk']:
                        if k == 'phoneme_dict':
                            _state[k] = load_phoneme_dict(*_state[k])
                        if not _state[k] == vars(self.state)[k]:
                            return False
                return True

    def _write_info(self):
        with open(os.path.join(self.state.dataset_path, self.info_filename), "w") as f:
            json.dump(vars(self.state), f)
        # with open(os.path.join(self.state.dataset_path, self.sample_index_filename), "w") as f:

        if len(self.sample_index[0]) == 3:
            filenames = {filename: file_idx
                         for file_idx, filename in enumerate(Counter([filename
                                                                      for _, filename, _ in
                                                                      self.sample_index]).keys())}

            _sample_index = np.array([(chunk_idx, filenames[filename], sample_idx)
                                      for chunk_idx, filename, sample_idx in self.sample_index], dtype=np.int32)

        elif len(self.sample_index[0]) == 4:
            filenames = {filename: file_idx
                         for file_idx, filename in enumerate(Counter([filename
                                                                      for _, filename, _, _ in
                                                                      self.sample_index]).keys())}

            _sample_index = np.array([(chunk_idx, filenames[filename], start_idx, end_idx)
                                      for chunk_idx, filename, start_idx, end_idx in self.sample_index], dtype=np.int32)
        else:
            raise RuntimeError

        filenames = {filename: file_idx for file_idx, filename in filenames.items()}
        np.savez(os.path.join(self.state.dataset_path, self.sample_index_filename),
                 filenames=filenames,
                 sample_index=_sample_index)

    def _read_info(self):
        with open(os.path.join(self.state.dataset_path, self.info_filename), "r") as f:
            _state = json.load(f)
            self.state = SimpleNamespace(**_state)
            self.state.phoneme_dict = load_phoneme_dict(*_state['phoneme_dict'])
        # with open(os.path.join(self.state.dataset_path, self.sample_index_filename), "r") as f:
        # _sample_index = json.load(f)
        _sample_index = np.load(os.path.join(self.state.dataset_path, self.sample_index_filename))

        if 'filenames' in _sample_index:
            self.sample_index = {}
            self.sample_index['filenames'] = _sample_index['filenames'].item()
            self.sample_index['sample_index'] = _sample_index['sample_index']
        else:
            raise RuntimeError


def apply_context(sample, start_idx, end_idx, context_left, context_right, aligned_labels):
    """
    Remove labels left and right to account for the needed context.

    Note:
        Reasons to just concatinate the context:

        Pro:

        - Like in production, we continously predict a frame with context
        - one frame and context corresponds to one out value, no confusion
        - MLP possible
        - easier to reason about
        - less confusion with wired effects of padding etc

        Contra:

        - recomputation of convolutions
        - not clear how to do it continously
        - more memory needed since it grows exponentially with the context size

    """

    lables = {}
    for label_name in sample['labels']:
        if aligned_labels[label_name]:
            assert end_idx > 0
            lables[label_name] = sample['labels'][label_name][start_idx: end_idx]

        else:
            lables[label_name] = sample['labels'][label_name]

    features = {}
    for feature_name in sample['features']:
        if all([not aligned_labels[label_name] for label_name in sample['labels']]):
            assert len(sample['features'][feature_name]) == end_idx - start_idx + context_left + context_right, \
                f"{len(sample['features'][feature_name])} {end_idx} {start_idx} {context_left} {context_right}"
        features[feature_name] = \
            apply_context_single_feat(
                sample['features'][feature_name],
                context_left, context_right, start_idx, end_idx)

    return features, lables


def np_dict_to_torch(_dict):
    for k in _dict:
        _dict[k] = torch.from_numpy(_dict[k])
    return _dict
