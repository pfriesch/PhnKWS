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

from data.data_util import load_features, split_chunks, load_labels, splits_by_seqlen, apply_context_single_feat, \
    filter_by_seqlen
from data.phoneme_dict import load_phoneme_dict, get_phoneme_dict, PhonemeDict
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
        self.overfit_small_batch = overfit_small_batch
        if isinstance(phoneme_dict, str) and os.path.exists(phoneme_dict):
            if 'lab_phn' in label_dict:
                # e2e ctc
                phoneme_dict = get_phoneme_dict(phoneme_dict,
                                                stress_marks=False, word_position_dependency=False)
            else:
                # framewise
                phoneme_dict = get_phoneme_dict(phoneme_dict,
                                                stress_marks=True, word_position_dependency=True)

        if not isinstance(phoneme_dict, PhonemeDict):
            raise ValueError(f"Got {phoneme_dict} for phoneme_dict, expected path to \"phones.txt\" or PhonemeDict")

        self.aligned_labels = {}

        if max_sample_len < 0:
            max_sample_len = float("inf")

        for label_name in label_dict:
            if label_dict[label_name]["label_opts"] == "ali-to-phones --per-frame=true" or \
                    label_dict[label_name]["label_opts"] == "ali-to-pdf":
                self.aligned_labels[label_name] = True

            elif label_dict[label_name]["label_opts"] == "ali-to-phones":
                self.aligned_labels[label_name] = False
                if max_seq_len < 0:
                    max_seq_len = None

            else:
                raise NotImplementedError

        self.shuffle_frames = shuffle_frames
        if self.shuffle_frames:
            assert max_seq_len is False or \
                   max_seq_len is None
            assert max_sample_len is False or \
                   max_sample_len is None

        self.data_cache_root = os.path.expanduser(data_cache_root)
        self.chunk_size = 100  # 1000 for TIMIT & 100 for libri
        self.samples_per_chunk = None
        self.max_len_per_chunk = None
        self.min_len_per_chunk = None
        self.cached_pt = 0
        self.max_seq_len = max_seq_len
        self.max_sample_len = max_sample_len
        self.min_sample_len = 0  # TODO ?
        self.max_label_length = max_label_length
        self.left_context = left_context
        self.right_context = right_context
        self.phoneme_dict = phoneme_dict
        self.label_index_from = 1  # TODO ?

        self.normalize_features = normalize_features

        self.dataset_path = os.path.join(self.data_cache_root, self.dataset_prefix, "processed", dataset_name)
        if not self._check_exists(feature_dict, label_dict):
            self._convert_from_kaldi_format(feature_dict, label_dict)
        if not self._check_exists(feature_dict, label_dict):
            raise RuntimeError('Dataset not found.')
        self._read_info()

        self.cached_samples = torch.load(
            os.path.join(self.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

    def _getitem_shuffled_frames(self, index):
        # Reason for not just doing a big matrix with all samples and skipping the following mess:
        # * Since we want to save disk space it is not feasible to precompute the context windows and append
        #   them to the samples and save this. Therefore we have to add the context here while loading.
        # * We also want to know the total number of frames in the dataset to return a length.

        # Get the chunk the index is in
        _samples_per_chunk_cumulative = np.cumsum(self.samples_per_chunk)
        chunk_idx = bisect.bisect_right(_samples_per_chunk_cumulative, index)
        assert _samples_per_chunk_cumulative[chunk_idx - 1] <= index < _samples_per_chunk_cumulative[
            chunk_idx] or 0 <= index < _samples_per_chunk_cumulative[0], \
            f"{_samples_per_chunk_cumulative[chunk_idx - 1]} <= {index} <" \
            + f" {_samples_per_chunk_cumulative[chunk_idx]} or 0 < {index} < {_samples_per_chunk_cumulative[0]}"

        if self.cached_pt != chunk_idx:
            self.cached_pt = chunk_idx
            self.cached_samples = torch.load(
                os.path.join(self.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

        # get the file the index is in
        in_chunk_dx = index - (_samples_per_chunk_cumulative[chunk_idx - 1] if chunk_idx > 0 else 0)
        filenames, end_idx_total_in_chunk = self.cached_samples['sample_splits']
        assert in_chunk_dx <= end_idx_total_in_chunk[-1], f"{in_chunk_dx} <= {end_idx_total_in_chunk[-1]}"
        file_index = bisect.bisect_right(end_idx_total_in_chunk, in_chunk_dx)

        # get the actual sample frame from the file
        filename = filenames[file_index]
        in_sample_index = in_chunk_dx - (end_idx_total_in_chunk[file_index - 1] if file_index > 0 else 0)
        # in_sample_index is the length if the sample instead of the index till here
        # so to get the index it's len - 1
        assert 0 <= in_sample_index
        assert in_sample_index < end_idx_total_in_chunk[file_index]

        sample = self.cached_samples['samples'][filename]
        lables = {}
        for label_name in sample['labels']:
            lables[label_name] = np.expand_dims(sample['labels'][label_name][in_sample_index], 0)

        features = {}
        for feature_name in sample['features']:
            features[feature_name] = \
                np.expand_dims(sample['features'][feature_name][
                               in_sample_index:
                               in_sample_index + self.left_context + self.right_context + 1, :].T, 0)

            assert features[feature_name].shape[2] == self.left_context + self.right_context + 1
            assert end_idx_total_in_chunk[file_index] - (
                end_idx_total_in_chunk[file_index - 1] if file_index > 0 else 0) == len(
                sample['features'][feature_name]) - self.left_context - self.right_context
            assert 0 <= in_sample_index + self.right_context + 1 < len(
                sample['features'][feature_name]), \
                "{} <!= {}".format(in_sample_index + self.left_context + self.right_context + 1,
                                   len(sample['features'][feature_name]))

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

        _samples_per_chunk_cumulative = np.cumsum(self.samples_per_chunk)
        chunk_idx = bisect.bisect_right(_samples_per_chunk_cumulative, index)
        if not (_samples_per_chunk_cumulative[chunk_idx - 1] <= index <
                _samples_per_chunk_cumulative[chunk_idx] or 0 <= index < _samples_per_chunk_cumulative[0]):
            assert _samples_per_chunk_cumulative[chunk_idx - 1] <= index < _samples_per_chunk_cumulative[
                chunk_idx] or 0 <= index < _samples_per_chunk_cumulative[0], \
                f"{_samples_per_chunk_cumulative[chunk_idx - 1]} <= {index} < " \
                + f"{_samples_per_chunk_cumulative[chunk_idx]} or 0 <= {index} < {_samples_per_chunk_cumulative[0]}"

        if self.cached_pt != chunk_idx:
            self.cached_pt = chunk_idx
            self.cached_samples = torch.load(
                os.path.join(self.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

        # get the file the index is in
        in_chunk_dx = index - (_samples_per_chunk_cumulative[chunk_idx - 1] if chunk_idx > 0 else 0)
        assert in_chunk_dx < self.samples_per_chunk[chunk_idx]

        filename, start_idx, end_idx = self.cached_samples['sample_splits'][in_chunk_dx]

        features, lables = apply_context(self.cached_samples['samples'][filename],
                                         start_idx=start_idx, end_idx=end_idx,
                                         context_right=self.right_context, context_left=self.left_context,
                                         aligned_labels=self.aligned_labels)
        for feature_name in features:
            if not end_idx - start_idx == len(features[feature_name]):
                assert end_idx - start_idx == len(features[feature_name]), \
                    f"{end_idx - start_idx} =!= {len(features[feature_name])}"

        return filename, features, lables

    def __getitem__(self, index):
        if self.shuffle_frames:
            filename, features, lables = self._getitem_shuffled_frames(index)

        else:
            filename, features, lables = self._getitem_sequential(index)

        if self.normalize_features:
            # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
            for feature_name in features:
                features[feature_name] = (features[feature_name] -
                                          np.expand_dims(self.cached_samples['means'][feature_name], axis=-1)) / \
                                         np.expand_dims(self.cached_samples['std'][feature_name], axis=-1)

        return filename, np_dict_to_torch(features), np_dict_to_torch(lables)

    def __len__(self):
        return sum(self.samples_per_chunk)

    def _load_labels(self, label_dict, label_index_from, max_label_length):
        all_labels_loaded = {}

        for lable_name in label_dict:
            all_labels_loaded[lable_name] = load_labels(label_dict[lable_name]['label_folder'],
                                                        label_dict[lable_name]['label_opts'])

            if max_label_length > 0 and max_label_length is not None:
                all_labels_loaded[lable_name] = \
                    {l: all_labels_loaded[lable_name][l] for l in all_labels_loaded[lable_name]
                     if len(all_labels_loaded[lable_name][l]) < max_label_length}

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

            max_label = max([all_labels_loaded[lable_name][l].max() for l in all_labels_loaded[lable_name]])
            min_label = min([all_labels_loaded[lable_name][l].min() for l in all_labels_loaded[lable_name]])
            logger.debug(
                f"Max label: {max_label}")
            logger.debug(
                f"min label: {min_label}")

            if min_label > 0:
                logger.warn(f"label {lable_name} does not seem to be indexed from 0 -> making it indexed from 0")
                for l in all_labels_loaded[lable_name]:
                    all_labels_loaded[lable_name][l] = all_labels_loaded[lable_name][l] - 1

                max_label = max([all_labels_loaded[lable_name][l].max() for l in all_labels_loaded[lable_name]])
                min_label = min([all_labels_loaded[lable_name][l].min() for l in all_labels_loaded[lable_name]])
                logger.debug(
                    f"Max label new : {max_label}")
                logger.debug(
                    f"min label new: {min_label}")

            if label_index_from != 0:
                assert label_index_from > 0
                all_labels_loaded[lable_name] = {filename:
                                                     all_labels_loaded[lable_name][filename] + label_index_from
                                                 for filename in all_labels_loaded[lable_name]}

        return all_labels_loaded

    def _filter_samples_by_length(self, file_names, feature_dict, features_loaded, label_dict, all_labels_loaded):

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

        return samples

    def _make_frames_shuffled(self, samples_list, main_feat):
        # framewise shuffled frames
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

        return sample_splits

    def _make_frames_sequential(self, samples_list, main_feat, chnk_id):
        # sequential data
        if any([not self.aligned_labels[label_name] for label_name in self.aligned_labels]):
            assert all([not self.aligned_labels[label_name] for label_name in self.aligned_labels])
            # unaligned labels
            sample_splits, min_len = filter_by_seqlen(samples_list, self.max_seq_len,
                                                      self.left_context, self.right_context)
            logger.info(f"Used samples {len(sample_splits)}/{len(samples_list)} "
                        + f"for a max seq length of {self.max_seq_len} (min length was {min_len})")

        elif any([not self.aligned_labels[label_name] for label_name in self.aligned_labels]) \
                and not self.max_seq_len:
            assert all([not self.aligned_labels[label_name] for label_name in self.aligned_labels])
            # unaligned labels but no max_seq_len
            sample_splits = [
                (filename, self.left_context, len(sample_dict["features"][main_feat]) - self.right_context)
                for filename, sample_dict in samples_list]
        else:
            # framewise sequential
            if self.max_seq_len:
                sample_splits = splits_by_seqlen(samples_list, self.max_seq_len,
                                                 self.left_context, self.right_context)
            else:
                raise NotImplementedError("Framewise without max_seq_len not impl")

        for sample_id, start_idx, end_idx in sample_splits:
            self.max_len_per_chunk[chnk_id] = (end_idx - start_idx) \
                if (end_idx - start_idx) > self.max_len_per_chunk[chnk_id] else self.max_len_per_chunk[
                chnk_id]

            self.min_len_per_chunk[chnk_id] = (end_idx - start_idx) \
                if (end_idx - start_idx) < self.min_len_per_chunk[chnk_id] else self.min_len_per_chunk[
                chnk_id]

        # sort sigs/labels: longest -> shortest
        sample_splits = sorted(sample_splits, key=lambda x: x[2] - x[1])
        self.samples_per_chunk.append(len(sample_splits))

        if not len(sample_splits) == self.samples_per_chunk[chnk_id]:
            assert len(sample_splits) == self.samples_per_chunk[chnk_id], \
                f"{len(sample_splits)} =!= {self.samples_per_chunk[chnk_id]}"

        return sample_splits

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

        all_labels_loaded = self._load_labels(label_dict, self.label_index_from, self.max_label_length)

        with open(feature_dict[main_feat]["feature_lst_path"], "r") as f:
            lines = f.readlines()
        feat_list = lines
        random.shuffle(feat_list)
        file_chunks = list(split_chunks(feat_list, self.chunk_size, self.overfit_small_batch))

        self.max_len_per_chunk = [0] * len(file_chunks)
        self.min_len_per_chunk = [sys.maxsize] * len(file_chunks)
        self.samples_per_chunk = []
        for chnk_id, file_chnk in tqdm(list(enumerate(file_chunks))):  # TODO do threaded
            file_names = [feat.split(" ")[0] for feat in file_chnk]

            chnk_prefix = os.path.join(self.dataset_path, "chunk_{:04d}".format(chnk_id))

            features_loaded = {}
            for feature_name in feature_dict:
                chnk_scp = chnk_prefix + "feats.scp"
                with open(chnk_scp, "w") as f:
                    f.writelines(file_chnk)

                features_loaded[feature_name] = load_features(chnk_scp, feature_dict[feature_name]["feature_opts"])
                os.remove(chnk_scp)

            samples = self._filter_samples_by_length(file_names, feature_dict, features_loaded, label_dict,
                                                     all_labels_loaded)

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
                sample_splits = self._make_frames_sequential(samples_list, main_feat, chnk_id)
            else:
                sample_splits = self._make_frames_shuffled(samples_list, main_feat)

            torch.save(
                {"samples": samples,
                 "sample_splits": sample_splits,
                 "means": mean,
                 "std": std},
                chnk_prefix + ".pyt"
            )
            # TODO add warning when files get too big -> choose different chunk size

        assert len(self) > 0, \
            f"No sample with a max seq length of {self.max_seq_len} in the dataset! " \
            + f"Try to choose a higher max seq length to start with"
        self._write_info(feature_dict, label_dict)
        logger.info('Done extracting kaldi features!')

    def _check_exists(self, feature_dict, label_dict):
        if not os.path.exists(os.path.join(self.dataset_path, self.info_filename)):
            return False
        else:
            with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
                _info = json.load(f)

                # TODO overfit
            if ("feature_dict" not in _info
                    or "label_dict" not in _info
                    or "max_sample_len" not in _info
                    or "max_label_length" not in _info
                    or "left_context" not in _info
                    or "right_context" not in _info
                    or "normalize_features" not in _info
                    or "phoneme_dict" not in _info
                    or "max_seq_len" not in _info
                    or "shuffle_frames" not in _info
                    or "overfit_small_batch" not in _info):
                return False

            if (feature_dict != _info["feature_dict"]
                    or label_dict != _info["label_dict"]
                    or self.max_sample_len != _info["max_sample_len"]
                    or self.max_label_length != _info["max_label_length"]
                    or self.left_context != _info["left_context"]
                    or self.right_context != _info["right_context"]
                    or self.normalize_features != _info["normalize_features"]
                    or self.phoneme_dict != load_phoneme_dict(*_info["phoneme_dict"])
                    or self.max_seq_len != _info["max_seq_len"]
                    or self.shuffle_frames != _info["shuffle_frames"]
                    or self.overfit_small_batch != _info["overfit_small_batch"]):

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
                       "max_label_length": self.max_label_length,
                       "left_context": self.left_context,
                       "right_context": self.right_context,
                       "normalize_features": self.normalize_features,
                       "phoneme_dict": self.phoneme_dict,
                       "max_seq_len": self.max_seq_len,
                       "shuffle_frames": self.shuffle_frames,
                       "overfit_small_batch": self.overfit_small_batch,
                       "feature_dict": feature_dict,
                       "label_dict": label_dict}, f)

    def _read_info(self):
        with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
            _info = json.load(f)
            self.samples_per_chunk = _info["samples_per_chunk"]
            self.max_len_per_chunk = _info["max_len_per_chunk"]
            self.min_len_per_chunk = _info["min_len_per_chunk"]
            self.max_seq_len = _info["max_seq_len"]
            assert self.chunk_size == _info["chunk_size"]
            self.max_sample_len = _info["max_sample_len"]
            self.max_label_length = _info["max_label_length"]
            self.left_context = _info["left_context"]
            self.right_context = _info["right_context"]
            self.normalize_features = _info["normalize_features"]
            self.phoneme_dict = load_phoneme_dict(*_info["phoneme_dict"])
            self.shuffle_frames = _info["shuffle_frames"]
            self.feature_dict = _info["feature_dict"]
            self.label_dict = _info["label_dict"]


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
