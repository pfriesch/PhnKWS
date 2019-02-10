from __future__ import print_function

import json
import os
import os.path
import random
import errno
import sys
import bisect

import numpy as np
import torch.utils.data as data
import torch
from tqdm import tqdm

from data.data_util import load_features, split_chunks, load_labels, splits_by_seqlen, apply_context_single_feat
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

    def __init__(self, cache_data_root,
                 dataset_name,
                 feature_dict,
                 label_dict,
                 device,
                 max_sample_len=1000,
                 left_context=0,
                 right_context=0,
                 normalize_features=True,
                 phoneme_dict=None,  # e.g. kaldi/egs/librispeech/s5/data/lang/phones.txt

                 split_files_max_seq_len=100,
                 shuffle_frames=False

                 ):
        self.aligned_labels = {}

        for label_name in label_dict:
            if label_dict[label_name]["label_opts"] == "ali-to-phones --per-frame=true" or \
                    label_dict[label_name]["label_opts"] == "ali-to-pdf":
                self.aligned_labels[label_name] = True

            if label_dict[label_name]["label_opts"] == "ali-to-phones":
                self.aligned_labels[label_name] = False
                assert split_files_max_seq_len is False or \
                       split_files_max_seq_len is None or split_files_max_seq_len < 1, \
                    "Can't split the files without aligned labels."

        self.shuffle_frames = shuffle_frames
        if self.shuffle_frames:
            assert split_files_max_seq_len is False or \
                   split_files_max_seq_len is None
            assert max_sample_len is False or \
                   max_sample_len is None

        self.cache_data_root = os.path.expanduser(cache_data_root)
        self.chunk_size = 100  # 1000 for TIMIT & 100 for libri
        self.samples_per_chunk = None
        self.max_len_per_chunk = None
        self.min_len_per_chunk = None
        self.cached_pt = 0
        self.split_files_max_sample_len = split_files_max_seq_len
        self.max_sample_len = max_sample_len
        self.min_sample_len = 0  # TODO ?
        self.left_context = left_context
        self.right_context = right_context
        self.phoneme_dict = phoneme_dict

        self.normalize_features = normalize_features

        self.dataset_path = os.path.join(self.cache_data_root, self.dataset_prefix, "processed", dataset_name)
        if not self._check_exists(feature_dict, label_dict):
            self._convert_from_kaldi_format(feature_dict, label_dict)
        if not self._check_exists(feature_dict, label_dict):
            raise RuntimeError('Dataset not found.')
        self._read_info()

        self.cached_samples = []
        for chunk_idx in range(len(self.samples_per_chunk)):
            self.cached_samples.append(dict_to_torch(
                torch.load(os.path.join(self.dataset_path, "chunk_{:04d}.pyt".format(chunk_idx))), device))
            assert len(self.cached_samples[-1]['sample_splits']) == self.samples_per_chunk[chunk_idx], \
                f"{len(self.cached_samples[-1]['sample_splits'])} =!= {self.samples_per_chunk[chunk_idx]}"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample (dict):
        """

        if self.shuffle_frames:
            # Reason for not just doing a big matrix with all samples and skipping the following mess:
            # * Since we want to save disk space it is not feasible to precompute the context windows and append
            #   them to the samples and save this. Therefore we have to add the context here while loading.
            # * We also want to know the total number of frames in the dataset to return a length.

            # Get the chunk the index is in
            _samples_per_chunk_cumulative = np.cumsum(self.samples_per_chunk)
            chunk_idx = bisect.bisect_left(_samples_per_chunk_cumulative, index)
            assert _samples_per_chunk_cumulative[chunk_idx - 1] <= index < _samples_per_chunk_cumulative[
                chunk_idx] or 0 <= index < _samples_per_chunk_cumulative[0]

            _cached_samples = self.cached_samples[chunk_idx]

            # get the file the index is in
            in_chunk_dx = index - (_samples_per_chunk_cumulative[chunk_idx - 1] if chunk_idx > 0 else 0)
            filenames, end_idx_total_in_chunk = _cached_samples['sample_splits']
            assert in_chunk_dx <= end_idx_total_in_chunk[-1]
            file_index = bisect.bisect_left(end_idx_total_in_chunk, in_chunk_dx)

            # get the actual sample frame from the file
            filename = filenames[file_index]
            in_sample_index = in_chunk_dx - (end_idx_total_in_chunk[file_index - 1] if file_index > 0 else 0)
            # in_sample_index is the length if the sample instead of the index till here
            # so to get the index it's len - 1
            in_sample_index -= 1

            sample = _cached_samples['samples'][filename]
            lables = {}
            for label_name in sample['labels']:
                lables[label_name] = sample['labels'][label_name][in_sample_index].unsqueeze(0)

            features = {}
            for feature_name in sample['features']:
                features[feature_name] = \
                    sample['features'][feature_name][
                    in_sample_index:
                    in_sample_index + self.left_context + self.right_context + 1, :].transpose().unsqueeze(0)

                assert features[feature_name].shape[2] == self.left_context + self.right_context + 1
                assert end_idx_total_in_chunk[file_index] - (
                    end_idx_total_in_chunk[file_index - 1] if file_index > 0 else 0) == len(
                    sample['features'][feature_name]) - self.left_context - self.right_context
                assert 0 <= in_sample_index + self.left_context + self.right_context + 1 < len(
                    sample['features'][feature_name]), \
                    "{} <!= {}".format(in_sample_index + self.left_context + self.right_context + 1,
                                       len(sample['features'][feature_name]))

        elif self.split_files_max_sample_len:
            _samples_per_chunk_cumulative = np.cumsum(self.samples_per_chunk)
            chunk_idx = bisect.bisect_left(_samples_per_chunk_cumulative, index)
            assert _samples_per_chunk_cumulative[chunk_idx - 1] < index <= _samples_per_chunk_cumulative[
                chunk_idx] or 0 < index <= _samples_per_chunk_cumulative[0], \
                f"{_samples_per_chunk_cumulative[chunk_idx - 1]} < {index} < " \
                + f"{_samples_per_chunk_cumulative[chunk_idx]} or 0 < {index} < {_samples_per_chunk_cumulative[0]}"

            _cached_samples = self.cached_samples[chunk_idx]
            assert len(_cached_samples['sample_splits']) == self.samples_per_chunk[chunk_idx]

            # get the file the index is in
            in_chunk_dx = index - (_samples_per_chunk_cumulative[chunk_idx - 1] if chunk_idx > 0 else 0)
            in_chunk_dx -= 1  # TODO figure out this thing
            assert in_chunk_dx < self.samples_per_chunk[chunk_idx]

            filename, start_idx, end_idx = _cached_samples['sample_splits'][in_chunk_dx]

            features, lables = apply_context(_cached_samples['samples'][filename], context_right=self.right_context,
                                             context_left=self.left_context, aligned_labels=self.aligned_labels)
            narrow_by_split(features, lables, start_idx - self.left_context, end_idx - self.right_context)
            for feature_name in features:
                assert end_idx - start_idx == len(features[feature_name])
        else:
            raise NotImplementedError
        if self.normalize_features:
            # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
            for feature_name in features:
                features[feature_name] = (features[feature_name] -
                                          _cached_samples['means'][feature_name].unsqueeze(-1)) / \
                                         _cached_samples['std'][feature_name].unsqueeze(-1)

        return filename, features, lables

    def __len__(self):
        return sum(self.samples_per_chunk)

    def _check_exists(self, feature_dict, label_dict):
        if not os.path.exists(os.path.join(self.dataset_path, self.info_filename)):
            return False
        else:
            with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
                _info = json.load(f)

            if ("feature_dict" not in _info
                    or "label_dict" not in _info
                    or "max_sample_len" not in _info
                    or "left_context" not in _info
                    or "right_context" not in _info
                    or "normalize_features" not in _info
                    or "phoneme_dict" not in _info
                    or "split_files_max_sample_len" not in _info
                    or "shuffle_frames" not in _info):
                return False

            if (feature_dict != _info["feature_dict"]
                    or label_dict != _info["label_dict"]
                    or self.max_sample_len != _info["max_sample_len"]
                    or self.left_context != _info["left_context"]
                    or self.right_context != _info["right_context"]
                    or self.normalize_features != _info["normalize_features"]
                    or self.phoneme_dict != _info["phoneme_dict"]
                    or self.split_files_max_sample_len != _info["split_files_max_sample_len"]
                    or self.shuffle_frames != _info["shuffle_frames"]):

                return False
            else:
                return True

    def _write_info(self, feature_dict, label_dict):
        with open(os.path.join(self.dataset_path, self.info_filename), "w") as f:
            json.dump({"samples_per_chunk": self.samples_per_chunk,
                       "max_len_per_chunk": self.max_len_per_chunk,
                       "min_len_per_chunk": self.min_len_per_chunk,
                       "chunk_size": self.chunk_size,
                       "max_sample_len": self.max_sample_len,
                       "left_context": self.left_context,
                       "right_context": self.right_context,
                       "normalize_features": self.normalize_features,
                       "phoneme_dict": self.phoneme_dict,
                       "split_files_max_sample_len": self.split_files_max_sample_len,
                       "shuffle_frames": self.shuffle_frames,
                       "feature_dict": feature_dict,
                       "label_dict": label_dict}, f)

    def _read_info(self):
        with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
            _info = json.load(f)
            self.samples_per_chunk = _info["samples_per_chunk"]
            self.max_len_per_chunk = _info["max_len_per_chunk"]
            self.min_len_per_chunk = _info["min_len_per_chunk"]
            self.split_files_max_sample_len = _info["split_files_max_sample_len"]
            assert self.chunk_size == _info["chunk_size"]
            self.max_sample_len = _info["max_sample_len"]
            self.left_context = _info["left_context"]
            self.right_context = _info["right_context"]
            self.normalize_features = _info["normalize_features"]
            self.phoneme_dict = _info["phoneme_dict"]
            self.shuffle_frames = _info["shuffle_frames"]
            self.feature_dict = _info["feature_dict"]
            self.label_dict = _info["label_dict"]

    def _load_labels(self, label_dict):
        all_labels_loaded = {}

        for lable_name in label_dict:
            all_labels_loaded[lable_name] = load_labels(label_dict[lable_name]['label_folder'],
                                                        label_dict[lable_name]['label_opts'])

            if lable_name == "lab_phn":
                if self.phoneme_dict is not None:
                    for sample_id in all_labels_loaded[lable_name]:
                        assert max(all_labels_loaded[lable_name][sample_id]) <= max(
                            self.phoneme_dict.idx2reducedIdx.keys()), \
                            "Are you sure you have the righ phoneme dict?" + \
                            " Labels have higher indices than phonemes ( {} <!= {} )".format(
                                max(all_labels_loaded[lable_name][sample_id]),
                                max(self.phoneme_dict.idx2reducedIdx.keys()))

                        # map labels according to phoneme dict
                        tmp_labels = np.copy(all_labels_loaded[lable_name][sample_id])
                        for k, v in self.phoneme_dict.idx2reducedIdx.items():
                            tmp_labels[all_labels_loaded[lable_name][sample_id] == k] = v

                        all_labels_loaded[lable_name][sample_id] = tmp_labels

        return all_labels_loaded

    def _convert_from_kaldi_format(self, feature_dict, label_dict):
        main_feat = next(iter(feature_dict))

        # download files
        try:
            os.makedirs(self.dataset_path)
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
        file_chunks = list(split_chunks(feat_list, self.chunk_size))

        self.max_len_per_chunk = [0] * len(file_chunks)
        self.min_len_per_chunk = [sys.maxsize] * len(file_chunks)
        self.samples_per_chunk = []
        for chnk_id, file_chnk in tqdm(list(enumerate(file_chunks))):
            file_names = [feat.split(" ")[0] for feat in file_chnk]

            chnk_prefix = os.path.join(self.dataset_path, "chunk_{:04d}".format(chnk_id))

            features_loaded = {}
            for feature_name in feature_dict:
                chnk_scp = chnk_prefix + "feats.scp"
                with open(chnk_scp, "w") as f:
                    f.writelines(file_chnk)

                features_loaded[feature_name] = load_features(chnk_scp, feature_dict[feature_name]["feature_opts"])
                os.remove(chnk_scp)

            samples = {}

            for file in file_names:

                _continue = False
                for feature_name in feature_dict:
                    if file not in features_loaded[feature_name]:
                        logger.info("Skipping {}, not in features".format(file))
                        _continue = True
                        break
                for label_name in label_dict:
                    if file not in all_labels_loaded[label_name]:
                        logger.info("Skipping {}, not in labels".format(file))
                        _continue = True
                        break
                for feature_name in feature_dict:
                    if type(self.max_sample_len) == int and \
                            len(features_loaded[feature_name][file]) > self.max_sample_len:
                        logger.info("Skipping {}, feature of size {} too big ( {} expected) ".format(
                            file, len(features_loaded[feature_name][file]), self.max_sample_len))
                        _continue = True
                        break
                    if type(self.min_sample_len) == int and \
                            self.min_sample_len > len(features_loaded[feature_name][file]):
                        logger.info("Skipping {}, feature of size {} too small ( {} expected) ".format(
                            file, len(features_loaded[feature_name][file]), self.max_sample_len))
                        _continue = True
                        break

                if _continue:
                    continue

                samples[file] = {"features": {}, "labels": {}}
                for feature_name in feature_dict:
                    samples[file]["features"][feature_name] = features_loaded[feature_name][file]

                for label_name in label_dict:
                    samples[file]["labels"][label_name] = all_labels_loaded[label_name][file]

            samples_list = list(samples.items())

            mean = {}
            std = {}
            for feature_name in feature_dict:
                feat_concat = []
                for file in file_names:
                    feat_concat.append(features_loaded[feature_name][file])

                feat_concat = np.concatenate(feat_concat)
                mean[feature_name] = np.mean(feat_concat, axis=0)
                std[feature_name] = np.std(feat_concat, axis=0)

            if not self.shuffle_frames:
                if self.split_files_max_sample_len:
                    sample_splits = splits_by_seqlen(samples_list, self.split_files_max_sample_len,
                                                     self.left_context, self.right_context)

                else:
                    sample_splits = [
                        (filename, self.left_context, len(sample_dict["features"][main_feat]) - self.right_context)
                        for filename, sample_dict in samples_list]

                for sample_id, start_idx, end_idx in sample_splits:
                    self.max_len_per_chunk[chnk_id] = (end_idx - start_idx) \
                        if (end_idx - start_idx) > self.max_len_per_chunk[chnk_id] else self.max_len_per_chunk[chnk_id]

                    self.min_len_per_chunk[chnk_id] = (end_idx - start_idx) \
                        if (end_idx - start_idx) < self.min_len_per_chunk[chnk_id] else self.min_len_per_chunk[chnk_id]

                # sort sigs/labels: longest -> shortest
                sample_splits = sorted(sample_splits, key=lambda x: x[2] - x[1])
                self.samples_per_chunk.append(len(sample_splits))

            else:
                prev_index = 0
                samples_idices = []
                sample_ids = []
                # sample_id, end_idx_total
                for sample_id, data in samples_list:
                    prev_index += len(data['features'][main_feat]) - self.left_context - self.right_context

                    sample_ids.append(sample_id)
                    samples_idices.append(prev_index)

                sample_splits = (sample_ids, samples_idices)
                self.samples_per_chunk.append(samples_idices[-1])

            assert len(sample_splits) == self.samples_per_chunk[chnk_id]

            torch.save(
                {"samples": samples,
                 "sample_splits": sample_splits,
                 "means": mean,
                 "std": std},
                chnk_prefix + ".pyt"
            )
            # TODO add warning when files get too big -> choose different chunk size

        self._write_info(feature_dict, label_dict)
        logger.info('Done extracting kaldi features!')


def apply_context(sample, context_left, context_right, aligned_labels):
    """
    Remove labels left and right to account for the needed context.

    Note:
        Reasons to just concatinate the context:

        Pro:

        - Like in production, we continously predict a frame with context
        - one frame and context corresponds to one out value, no confusion
        - TDNN possible
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
            if context_right > 0:
                lables[label_name] = sample['labels'][label_name][context_left: -context_right]
            else:
                lables[label_name] = sample['labels'][label_name][context_left:]

        else:
            lables[label_name] = sample['labels'][label_name]

    features = {}
    for feature_name in sample['features']:
        features[feature_name] = \
            apply_context_single_feat(
                sample['features'][feature_name],
                context_left, context_right)

    return features, lables


def narrow_by_split(features, lables, start_idx, end_idx):
    for label_name in lables:
        lables[label_name] = lables[label_name][start_idx: end_idx]

    for feature_name in features:
        features[feature_name] = features[feature_name][start_idx: end_idx]


def dict_to_torch(_dict, device):
    for filename in _dict['samples']:
        for feat_name in _dict['samples'][filename]['features']:
            _dict['samples'][filename]['features'][feat_name] = torch.tensor(
                _dict['samples'][filename]['features'][feat_name], device=device)
        for lab_name in _dict['samples'][filename]['labels']:
            _dict['samples'][filename]['labels'][lab_name] = torch.tensor(
                _dict['samples'][filename]['labels'][lab_name], device=device)
    for feat_name in _dict['means']:
        _dict['means'][feat_name] = torch.tensor(_dict['means'][feat_name], device=device)
    for feat_name in _dict['std']:
        _dict['std'][feat_name] = torch.tensor(_dict['std'][feat_name], device=device)
    return _dict
