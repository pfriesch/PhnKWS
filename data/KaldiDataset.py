from __future__ import print_function

import json
import os
import os.path
import random
import errno
import sys

import numpy as np

import torch.utils.data as data
import torch
from tqdm import tqdm

from data.data_util import load_features, split_chunks, load_labels, splits_by_seqlen, apply_context_single_feat
from data.phoneme_dicts.phoneme_dict import get_phoneme_dict
from utils.logger_config import logger, Logger


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
            lables[label_name] = sample['labels'][label_name][context_left: -context_right]
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
                 max_sample_len=1000,
                 left_context=5,
                 right_context=5,
                 normalize_features=True,
                 phoneme_dict=None,  # e.g. kaldi/egs/librispeech/s5/data/lang/phones.txt

                 split_files_max_seq_len=100,

                 ):
        self.aligned_labels = {}
        #### sanity checks
        for label_name in label_dict:
            if label_dict[label_name]["label_opts"] == "ali-to-phones --per-frame=true" or \
                    label_dict[label_name]["label_opts"] == "ali-to-pdf":
                self.aligned_labels[label_name] = True

            if label_dict[label_name]["label_opts"] == "ali-to-phones":
                self.aligned_labels[label_name] = False
                assert split_files_max_seq_len == False or split_files_max_seq_len is None or split_files_max_seq_len < 1, \
                    "Can't split the files without aligned labels."
        #### / sanity checks

        self.cache_data_root = os.path.expanduser(cache_data_root)
        self.chunk_size = 100  # 1000 for TIMIT & 100 for libri
        self.samples_per_chunk = None
        self.max_len_per_chunk = None
        self.min_len_per_chunk = None
        self.cached_pt = 0
        self.split_files_max_seq_len = split_files_max_seq_len
        self.max_sample_len = max_sample_len
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

        self.samples = torch.load(
            os.path.join(self.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))

    def set_split_files_max_seq_len(self, split_files_max_seq_len):
        if self.split_files_max_seq_len != split_files_max_seq_len:
            self.split_files_max_seq_len = split_files_max_seq_len
            self._convert_from_kaldi_format(self.feature_dict, self.label_dict)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample (dict):
        """

        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.samples_list = torch.load(
                os.path.join(self.dataset_path, "chunk_{:04d}.pyt".format(self.cached_pt)))
        index = index % self.chunk_size
        filename, start_idx, end_idx = self.samples['sample_splits'][index]
        features, lables = apply_context(self.samples['samples'][filename], context_right=self.right_context,
                                         context_left=self.left_context, aligned_labels=self.aligned_labels)
        narrow_by_split(features, lables, start_idx - self.left_context, end_idx - self.right_context)
        for feature_name in features:
            assert end_idx - start_idx == len(features[feature_name])
        if self.normalize_features:
            # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
            for feature_name in features:
                features[feature_name] = (features[feature_name] -
                                          np.expand_dims(self.samples['means'][feature_name], axis=-1)) / \
                                         np.expand_dims(self.samples['sts'][feature_name], axis=-1)

        return filename, features, lables

    def __len__(self):
        return sum(self.samples_per_chunk)

    def _check_exists(self, feature_dict, label_dict):
        if not os.path.exists(os.path.join(self.dataset_path, self.info_filename)):
            return False
        else:
            with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
                _info = json.load(f)

            if not (feature_dict == _info["feature_dict"]
                    or label_dict == _info["label_dict"]
                    or self.split_files_max_seq_len == _info["feats_chunked_by_seqlen"]):
                return False
            else:
                return True

    def _write_info(self, feature_dict, label_dict):
        with open(os.path.join(self.dataset_path, self.info_filename), "w") as f:
            json.dump({"samples_per_chunk": self.samples_per_chunk,
                       "max_len_per_chunk": self.max_len_per_chunk,
                       "min_len_per_chunk": self.min_len_per_chunk,
                       "chunk_size": self.chunk_size,
                       "split_files_max_seq_len": self.split_files_max_seq_len,
                       "feature_dict": feature_dict,
                       "label_dict": label_dict}, f)

    def _read_info(self):
        with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
            _info = json.load(f)
            self.samples_per_chunk = _info["samples_per_chunk"]
            self.max_len_per_chunk = _info["max_len_per_chunk"]
            self.min_len_per_chunk = _info["min_len_per_chunk"]
            self.split_files_max_seq_len = _info["split_files_max_seq_len"]
            assert self.chunk_size == _info["chunk_size"]
            self.feature_dict = _info["feature_dict"]
            self.label_dict = _info["label_dict"]

    def _load_labels(self, label_dict):
        all_labels_loaded = {}

        for lable_name in label_dict:
            # phn_mapping[lable_name]) #TODO
            all_labels_loaded[lable_name] = load_labels(label_dict[lable_name]['label_folder'],
                                                        label_dict[lable_name]['label_opts'])

            if lable_name == "lab_phn":
                if self.phoneme_dict is not None:
                    for sample_id in all_labels_loaded[lable_name]:
                        assert max(all_labels_loaded[lable_name][sample_id]) <= max(
                            self.phoneme_dict.idx2reducedIdx.keys()), \
                            "Are you sure you have the righ phoneme dict? Labels have higher indices than phonemes ( {} <!= {} )" \
                                .format(max(all_labels_loaded[lable_name][sample_id]),
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

            if self.split_files_max_seq_len:
                sample_splits = splits_by_seqlen(samples_list, self.split_files_max_seq_len,
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

            torch.save(
                {"samples": samples,
                 "sample_splits": sample_splits,
                 "means": mean,
                 "sts": std},
                chnk_prefix + ".pyt"
            )
            self.samples_per_chunk.append(len(sample_splits))
            # TODO add warning when files get too big -> choose different chunk size

        self._write_info(feature_dict, label_dict)
        logger.info('Done extracting kaldi features!')


def test1():
    dataset = KaldiDataset(cache_data_root="/mnt/data/pytorch-kaldi/data",
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": "apply-cmvn --utt2spk=ark:/mnt/data/libs/kaldi/egs/timit/s5/data/train/utt2spk  ark:/mnt/data/libs/kaldi/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_phn": {
                                   "label_folder": "/mnt/data/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones",
                                   "lab_count_file": "none",
                                   "lab_data_folder": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/",
                                   "lab_graph": "/mnt/data/libs/kaldi/egs/timit/s5/exp/tri3/graph",
                               }
                           },
                           max_sample_len=200,
                           left_context=10, right_context=2,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict("/mnt/data/libs/kaldi/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=False

                           )
    return dataset


def test2():
    dataset = KaldiDataset(cache_data_root="/mnt/data/pytorch-kaldi/data",
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": "apply-cmvn --utt2spk=ark:/mnt/data/libs/kaldi/egs/timit/s5/data/train/utt2spk  ark:/mnt/data/libs/kaldi/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_cd_phn": {
                                   "label_folder": "/mnt/data/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-pdf",
                                   "lab_count_file": "auto",
                                   "lab_data_folder": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/",
                                   "lab_graph": "/mnt/data/libs/kaldi/egs/timit/s5/exp/tri3/graph"
                               },
                               "lab_phn": {
                                   "label_folder": "/mnt/data/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones --per-frame=true",
                                   "lab_count_file": "none",
                                   "lab_data_folder": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/",
                                   "lab_graph": "/mnt/data/libs/kaldi/egs/timit/s5/exp/tri3/graph"
                               }
                           },
                           max_sample_len=100,
                           left_context=10, right_context=2,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict("/mnt/data/libs/kaldi/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=100

                           )
    return dataset


if __name__ == '__main__':
    logger.configure_logger(out_folder="/mnt/data/pytorch-kaldi/data")
    if os.path.exists("/mnt/data/pytorch-kaldi/data/kaldi/processed/TIMIT_tr/info.json"):
        os.remove("/mnt/data/pytorch-kaldi/data/kaldi/processed/TIMIT_tr/info.json")

    ds = test1()
    print("unaligned")
    print(len(ds))
    filename, features, lables = next(iter(ds))
    print(filename)
    for feature_name in features:
        print(feature_name, features[feature_name].shape)
    for label_name in lables:
        print(label_name, lables[label_name].shape)
    input()
    os.remove("/mnt/data/pytorch-kaldi/data/kaldi/processed/TIMIT_tr/info.json")
    ds = test2()
    print("force-aligned")
    print(len(ds))
    filename, features, lables = next(iter(ds))
    print(filename)
    for feature_name in features:
        print(feature_name, features[feature_name].shape)
    for label_name in lables:
        print(label_name, lables[label_name].shape)
