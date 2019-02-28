from __future__ import print_function

import json
import os
import os.path
import random
import errno
import sys
from functools import partial
from multiprocessing.pool import Pool
from types import SimpleNamespace

import numpy as np
import torch.utils.data as data
import torch
from tqdm import tqdm

from data.data_util import split_chunks, apply_context_single_feat
from data.kaldi_dataset_utils import convert_chunk_from_kaldi_format
from data.phoneme_dict import load_phoneme_dict, PhonemeDict
from utils.logger_config import logger
from data.datasets import DatasetType


# inspired by https://github.com/pytorch/audio/blob/master/torchaudio/datasets/vctk.py
class BaseKaldiDataset(data.Dataset):
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
                 dataset_type,
                 max_sample_len=None,
                 left_context=None,
                 right_context=None,
                 normalize_features=None,
                 aligned_labels=False,
                 max_seq_len=None,
                 max_label_length=None,
                 overfit_small_batch=False
                 ):

        self.state = SimpleNamespace()
        self.state.dataset_type = dataset_type

        self.state.overfit_small_batch = overfit_small_batch

        self.state.aligned_labels = aligned_labels

        if max_sample_len is None or max_sample_len < 0:
            max_sample_len = float("inf")

        self.state.data_cache_root = os.path.expanduser(data_cache_root)
        self.state.chunk_size = 100
        self.state.max_len_per_chunk = None
        self.state.min_len_per_chunk = None
        self.state.max_seq_len = max_seq_len
        self.state.max_sample_len = max_sample_len
        self.state.min_sample_len = left_context + right_context + 1  # TODO ?
        self.state.max_label_length = max_label_length
        self.state.left_context = left_context
        self.state.right_context = right_context
        self.state.label_index_from = 1  # TODO ?

        self.cached_pt = 0
        self.sample_index = []
        self.filename_index = []

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
        raise NotImplementedError

    @property
    def samples_per_chunk(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _load_labels(self, label_dict):
        raise NotImplementedError

    def _filenames_from_sample_index(self, sample_index):
        """

        :return: filenames, _sample_index
        """
        raise NotImplementedError

    def _getitem(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        filename, features, lables = self._getitem(index)

        if self.state.normalize_features:
            # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
            for feature_name in features:
                features[feature_name] = (features[feature_name] -
                                          np.expand_dims(self.cached_samples['means'][feature_name], axis=-1)) / \
                                         np.expand_dims(self.cached_samples['std'][feature_name], axis=-1)

        return filename, np_dict_to_torch(features), np_dict_to_torch(lables)

    def _check_labels_indexed_from(self, all_labels_loaded, label_name):

        max_label = max([all_labels_loaded[label_name][l].max() for l in all_labels_loaded[label_name]])
        min_label = min([all_labels_loaded[label_name][l].min() for l in all_labels_loaded[label_name]])
        logger.debug(
            f"Max label: {max_label}")
        logger.debug(
            f"min label: {min_label}")

        if min_label > 0:
            logger.warn(f"label {label_name} does not seem to be indexed from 0 -> making it indexed from 0")
            for l in all_labels_loaded[label_name]:
                all_labels_loaded[label_name][l] = all_labels_loaded[label_name][l] - 1

            max_label = max([all_labels_loaded[label_name][l].max() for l in all_labels_loaded[label_name]])
            min_label = min([all_labels_loaded[label_name][l].min() for l in all_labels_loaded[label_name]])
            logger.debug(
                f"Max label new : {max_label}")
            logger.debug(
                f"min label new: {min_label}")

        if self.state.label_index_from != 0:
            assert self.state.label_index_from > 0
            all_labels_loaded[label_name] = {filename:
                                                 all_labels_loaded[label_name][filename] + self.state.label_index_from
                                             for filename in all_labels_loaded[label_name]}

    def _convert_from_kaldi_format(self, feature_dict, label_dict):
        logger.info("Converting features from kaldi features!")
        main_feat = next(iter(feature_dict))

        try:
            os.makedirs(self.state.dataset_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        all_labels_loaded = self._load_labels(label_dict)

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
                                                   shuffle_frames=self.state.dataset_type \
                                                                  == DatasetType.FRAMEWISE_SHUFFLED_FRAMES,
                                                   main_feat=main_feat,
                                                   aligned_labels=self.state.aligned_labels,
                                                   max_sample_len=self.state.max_sample_len,
                                                   min_sample_len=self.state.min_sample_len,
                                                   max_seq_len=self.state.max_seq_len,
                                                   left_context=self.state.left_context,
                                                   right_context=self.state.right_context)

        _sample_index = []
        with tqdm(total=len(file_chunks)) as pbar:
            with Pool() as pool:
                chunksize = len(file_chunks) // (
                        2 * os.cpu_count())
                if chunksize < 1:
                    chunksize = 1
                for chnk_id, sample_splits, max_len, min_len in pool.imap_unordered(_convert_chunk_from_kaldi_format,
                                                                                    enumerate(file_chunks),
                                                                                    chunksize=chunksize):
                    _sample_index.extend(sample_splits)
                    if max_len is not None:
                        self.state.max_len_per_chunk[chnk_id] = max_len
                    if min_len is not None:
                        self.state.min_len_per_chunk[chnk_id] = min_len
                    pbar.update()
                    pbar.set_description(f"{chnk_id}")

        assert len(_sample_index) > 0, \
            f"No sample with a max seq length of {self.state.max_seq_len} in the dataset! " \
            + f"Try to choose a higher max seq length to start with"
        self._write_info(_sample_index)
        logger.info('Done extracting kaldi features!')

    def _check_exists(self):
        if not os.path.exists(os.path.join(self.state.dataset_path, self.info_filename)):
            return False
        else:
            with open(os.path.join(self.state.dataset_path, self.info_filename), "r") as f:
                _state = json.load(f, object_hook=custom_decode_json)
                if not set(vars(self.state).keys()) == set(_state.keys()):
                    return False
                for k, v in vars(self.state).items():
                    if k not in ['max_len_per_chunk', 'min_len_per_chunk']:
                        if not _state[k] == vars(self.state)[k]:
                            return False
                return True

    def _write_info(self, sample_index):
        with open(os.path.join(self.state.dataset_path, self.info_filename), "w") as f:
            json.dump(vars(self.state), f, cls=CustomJSONEncoder)

        filenames, _sample_index = self._filenames_from_sample_index(sample_index)

        filenames = {filename: file_idx for file_idx, filename in filenames.items()}
        np.savez(os.path.join(self.state.dataset_path, self.sample_index_filename),
                 filenames=filenames,
                 sample_index=_sample_index)

    def _read_info(self):
        with open(os.path.join(self.state.dataset_path, self.info_filename), "r") as f:
            _state = json.load(f, object_hook=custom_decode_json)
            self.state = SimpleNamespace(**_state)

        _sample_index = np.load(os.path.join(self.state.dataset_path, self.sample_index_filename))

        assert 'filenames' in _sample_index
        self.sample_index = {}
        self.filename_index = _sample_index['filenames'].item()
        self.sample_index = _sample_index['sample_index']


def apply_context_full_sequence(sample, start_idx, end_idx, context_left, context_right, aligned_labels):
    lables = {}
    for label_name in sample['labels']:
        if aligned_labels:
            assert end_idx > 0
            lables[label_name] = sample['labels'][label_name][start_idx: end_idx]
            # assert len(sample['labels'][label_name]) - context_left - context_right >= lables[label_name]

        else:
            lables[label_name] = sample['labels'][label_name]

    features = {}
    for feature_name in sample['features']:
        assert start_idx - context_left >= 0
        assert end_idx + context_right <= len(sample['features'][feature_name])
        features[feature_name] = np.expand_dims(
            sample['features'][feature_name][start_idx - context_left: end_idx + context_right], axis=-1)

    return features, lables


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
        if aligned_labels:
            assert end_idx > 0
            lables[label_name] = sample['labels'][label_name][start_idx: end_idx]

        else:
            lables[label_name] = sample['labels'][label_name]

    features = {}
    for feature_name in sample['features']:
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


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) == DatasetType:
            return {"DatasetType": str(obj)}
        # elif isinstance(obj, PhonemeDict):
        #     return {"DatasetType": str(obj)}

        return json.JSONEncoder.default(self, obj)


def custom_decode_json(d):
    if "DatasetType" in d:
        name, member = d["DatasetType"].split(".")
        return DatasetType[member]
    elif "phoneme_dict" in d:
        return load_phoneme_dict(*d["phoneme_dict"])
    else:
        return d
