from __future__ import print_function

import json
import os
import os.path
import random
import errno
import sys

import torch.utils.data as data
import torch

from data.data_util import load_features, split_chunks, load_labels, chunks_by_seqlen


# inspired by https://github.com/pytorch/audio/blob/master/torchaudio/datasets/vctk.py
class KaldiDataset(data.Dataset):
    dataset_prefix = "kaldi"
    info_filename = "info.json"

    def __init__(self, root,
                 dataset_name,
                 feature_dict,
                 label_dict,
                 chunk_by_seqlen=True,
                 max_seq_len=100,
                 left_context=5,
                 right_context=5,
                 normalize=True,

                 ):
        self.root = os.path.expanduser(root)
        self.chunk_size = 1000
        self.samples_per_chunk = None
        self.max_len_per_chunk = None
        self.min_len_per_chunk = None
        self.cached_pt = 0
        self.chunked_by_seqlen = chunk_by_seqlen
        self.max_seq_len = max_seq_len
        self.left_context = left_context
        self.right_context = right_context
        self.normalize = normalize

        self.dataset_path = os.path.join(self.root, self.dataset_prefix, "processed", dataset_name)

        self.convert_from_kaldi_format(feature_dict, label_dict)
        if not self._check_exists(feature_dict, label_dict):
            raise RuntimeError('Dataset not found.')
        self._read_info()

        self.samples, self.chunks = torch.load(
            os.path.join(self.dataset_path, "chunk_{:04d}_samples.pyt".format(self.cached_pt)))

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
                os.path.join(self.dataset_path, "chunk_{:04d}_samples.pyt".format(self.cached_pt)))
        index = index % self.chunk_size

        # if self.left_context > 0 or self.right_context > 0:
        #     samples_views_list = apply_context(samples_views_list, self.left_context, self.right_context)
        return self.samples_list[index]

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
                    or self.chunked_by_seqlen == _info["feats_chunked_by_seqlen"]):
                return False
            else:
                return True

    def _write_info(self, feature_dict, label_dict):
        with open(os.path.join(self.dataset_path, self.info_filename), "w") as f:
            json.dump({"samples_per_chunk": self.samples_per_chunk,
                       "max_len_per_chunk": self.max_len_per_chunk,
                       "min_len_per_chunk": self.min_len_per_chunk,
                       "chunk_size": self.chunk_size,
                       "chunked_by_seqlen": self.chunked_by_seqlen,
                       "feature_dict": feature_dict,
                       "label_dict": label_dict}, f)

    def _read_info(self):
        with open(os.path.join(self.dataset_path, self.info_filename), "r") as f:
            _info = json.load(f)
            self.samples_per_chunk = _info["samples_per_chunk"]
            self.max_len_per_chunk = _info["max_len_per_chunk"]
            self.min_len_per_chunk = _info["min_len_per_chunk"]
            assert self.chunk_size == _info["chunk_size"]
            self.feature_dict = _info["feature_dict"]
            self.label_dict = _info["label_dict"]

    def make_chunks(self, samples_list):

        for chnk_id in range(int(len(self) // self.chunk_size) + 1):

            # TODO assert framewise
            self.chunks = chunks_by_seqlen(samples_list, self.max_seq_len,
                                      self.left_context, self.right_context)

            for filename, start_idx, end_idx in self.chunks:
                # filename, sample_dict = samples[filename]
                # take the last features assuming all have the samme length #todo check that
                self.max_len_per_chunk[chnk_id] = (end_idx - start_idx) \
                    if (end_idx - start_idx) > self.max_len_per_chunk[chnk_id] else self.max_len_per_chunk[chnk_id]

                self.min_len_per_chunk[chnk_id] = (end_idx - start_idx) \
                    if (end_idx - start_idx) < self.min_len_per_chunk[chnk_id] else self.min_len_per_chunk[chnk_id]

    def convert_from_kaldi_format(self, feature_dict, label_dict):
        if self._check_exists(feature_dict, label_dict):
            return

        main_feat = next(iter(feature_dict))

        # download files
        try:
            os.makedirs(self.dataset_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        with open(feature_dict[main_feat]["feature_lst_path"], "r") as f:
            lines = f.readlines()
        feat_list = lines
        random.shuffle(feat_list)
        chunks = list(split_chunks(feat_list, self.chunk_size))

        self.max_len_per_chunk = [0] * len(chunks)
        self.min_len_per_chunk = [sys.maxsize] * len(chunks)
        self.samples_per_chunk = []
        for chnk_id, chnk in enumerate(chunks):
            file_names = [feat.split(" ")[0] for feat in chnk]

            chnk_prefix = os.path.join(self.dataset_path, "chunk_{:04d}_".format(chnk_id))

            features_loaded = {}
            for feature_name in feature_dict:
                chnk_scp = chnk_prefix + "feats.scp"
                with open(chnk_scp, "w") as f:
                    f.writelines(chnk)

                features_loaded[feature_name] = load_features(chnk_scp, feature_dict[feature_name]["feature_opts"])
                os.remove(chnk_scp)

            labels_loaded = {}

            for lable_name in label_dict:
                # phn_mapping[lable_name]) #TODO
                labels_loaded[lable_name] = load_labels(label_dict[lable_name]['label_folder'],
                                                        label_dict[lable_name]['label_opts'],
                                                        filenames=list(file_names))

            samples = {}

            for file in file_names:
                samples[file] = {"features": {}, "labels": {}}
                for feature_name in feature_dict:
                    samples[file]["features"][feature_name] = features_loaded[feature_name][file]

                for lable_name in label_dict:
                    samples[file]["labels"][lable_name] = labels_loaded[lable_name][file]

            samples_list = list(samples.items())

            # if self.chunked_by_seqlen:
            #     # TODO assert framewise
            #     chunks = chunks_by_seqlen(samples_list, self.max_seq_len,
            #                               self.left_context, self.right_context)

            # features_field = "features_context" if self.left_context + self.right_context > 0 else "features"

            # for filename, start_idx, end_idx in chunks:
            #     # filename, sample_dict = samples[filename]
            #     # take the last features assuming all have the samme length #todo check that
            #     self.max_len_per_chunk[chnk_id] = (end_idx - start_idx) \
            #         if (end_idx - start_idx) > self.max_len_per_chunk[chnk_id] else self.max_len_per_chunk[chnk_id]
            #
            #     self.min_len_per_chunk[chnk_id] = (end_idx - start_idx) \
            #         if (end_idx - start_idx) < self.min_len_per_chunk[chnk_id] else self.min_len_per_chunk[chnk_id]

            samples_list = list(samples.items())

            # sort sigs/labels: longest -> shortest
            chunks = sorted(chunks, key=lambda x: x[2] - x[1])

            torch.save(
                (samples, chunks),
                chnk_prefix + "samples.pyt"
            )
            self.samples_per_chunk.append(len(chunks))

        self._write_info(feature_dict, label_dict)
        print('Done!')

    # def norm(self):
    #     if label_dict is not None:
    #         for label_name in label_dict:
    #             if label_name == "lab_mono":
    #                 # Adding 1 to use 0 padding for framewise or 0 as blank with ctc
    #                 label_chunks[label_name] -= 1
    #
    #             label_chunks[label_name] += 1
    #
    #             if label_chunks[label_name].min() != 0:
    #                 logger.warn("got label with min {}".format(label_chunks[label_name].min()))
    #             if label_chunks[label_name].max() >= 48:
    #                 logger.warn("got label with max {} {}".format(label_chunks[label_name].max(), label_name))
    #     # if normalize_feat:
    #     #     for feature_name in feature_dict:
    #     #         feature_chunks[feature_name] = (feature_chunks[feature_name] - np.mean(feature_chunks[feature_name],
    #     #                                                                                axis=0)) / np.std(
    #     #             feature_chunks[feature_name], axis=0)


if __name__ == '__main__':
    dataset = KaldiDataset("/mnt/data/pytorch-kaldi/data",
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": "apply-cmvn --utt2spk=ark:/mnt/data/libs/kaldi/egs/timit/s5/data/train/utt2spk  ark:/mnt/data/libs/kaldi/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                                   "n_chunks": 5
                               }
                           },
                           label_dict={
                               "lab_cd": {
                                   "label_folder": "/mnt/data/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-pdf",
                                   "lab_count_file": "auto",
                                   "lab_data_folder": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/",
                                   "lab_graph": "/mnt/data/libs/kaldi/egs/timit/s5/exp/tri3/graph"
                               },
                               "lab_mono": {
                                   "label_folder": "/mnt/data/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones --per-frame=true",
                                   "lab_count_file": "none",
                                   "lab_data_folder": "/mnt/data/libs/kaldi/egs/timit/s5/data/train/",
                                   "lab_graph": "/mnt/data/libs/kaldi/egs/timit/s5/exp/tri3/graph"
                               }
                           })
    for elem in dataset:
        print("t")
