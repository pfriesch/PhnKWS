import itertools
import torch
from collections import OrderedDict

import numpy as np

from data import kaldi_io


def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


def split_chunks(seq, size, overfit_small_batch):
    assert len(seq) > 0
    if overfit_small_batch:
        seq = seq[:10]
        return [seq]
    newseq = []
    chunk = 0
    for chunk in range(len(seq) // size):
        newseq.append(seq[chunk * size:chunk * size + size])
    if len(seq[chunk * size + size:]) > 0:
        newseq.append(seq[chunk * size + size:])
    return newseq


def load_features(feature_lst_path, feature_opts):
    features_loaded = \
        {k: m for k, m in
         kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(feature_lst_path, feature_opts))}
    assert len(features_loaded) > 0

    return features_loaded


def load_labels(label_folder, label_opts):
    labels_loaded = \
        {k: v for k, v in
         kaldi_io.read_vec_int_ark(
             'gunzip -c {}/ali*.gz | {} {}/final.mdl ark:- ark:-|'
                 .format(label_folder, label_opts, label_folder))}
    assert len(labels_loaded) > 0
    return labels_loaded


def load_kws(feature_dict, label_dict, kw2phn_mapping):
    features_loaded = {}
    labels_loaded = {}

    for feature_name in feature_dict:
        feature_lst_path = feature_dict[feature_name]['feature_lst_path']
        feature_opts = feature_dict[feature_name]['feature_opts']

        features_loaded[feature_name] = \
            {k: m for k, m in
             kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(feature_lst_path, feature_opts))}
        assert len(features_loaded[feature_name]) > 0

    for label_name in label_dict:
        text_file = label_dict[label_name]['text_file']
        with open(text_file, "r") as f:
            lines = f.readlines()

        labels_loaded[label_name] = \
            {filenames: kw2phn_mapping[text]['phn_ids']
             for filenames, text in
             [l.strip().split(" ") for l in lines] if text in kw2phn_mapping}

    all_files = set(itertools.chain.from_iterable(
        [set(labels_loaded[l]) for l in labels_loaded] + [set(features_loaded[f]) for f in features_loaded]))
    all_files_intersect = set.intersection(
        *[set(labels_loaded[l]) for l in labels_loaded] + [set(features_loaded[f]) for f in features_loaded])
    print("removed {} files because of missing labels".format(len(all_files) - len(all_files_intersect)))
    for feature_name in feature_dict:
        features_loaded[feature_name] = {filename: features_loaded[feature_name][filename]
                                         for filename in features_loaded[feature_name]
                                         if filename in all_files_intersect}
    for label_name in label_dict:
        labels_loaded[label_name] = {filename: labels_loaded[label_name][filename]
                                     for filename in labels_loaded[label_name]
                                     if filename in all_files_intersect}

    return features_loaded, labels_loaded


def filter_by_seqlen(samples_list, max_sequence_length, context_left, context_right):
    # samples_list_splited = []
    filtered_sample_list = []

    num_removed = 0
    min_len = float('inf')

    for sample in samples_list:
        filename, sample_dict = sample

        assert len(sample_dict["features"]) == 1  # TODO multi feature
        for feature_name in sample_dict["features"]:
            if min_len > len(sample_dict["features"][feature_name]) - (
                    context_left + context_right):
                # just for logging purposes
                min_len = len(sample_dict["features"][feature_name]) - (
                        context_left + context_right)
            if len(sample_dict["features"][feature_name]) - (
                    context_left + context_right) <= max_sequence_length and max_sequence_length > 0:
                start_idx = context_left
                end_idx = len(sample_dict["features"][feature_name]) - context_right
                filtered_sample_list.append((filename, start_idx, end_idx))
            else:
                num_removed += 1

    return filtered_sample_list, min_len


def splits_by_seqlen(samples_list, max_sequence_length, context_left, context_right):
    # TODO remove with 1/4 of max length -> add to config
    # TODO add option weather the context_size is applied to the minimum sequence length
    min_sequence_length = max_sequence_length // 4  # + (context_left + context_right)

    # samples_list_splited = []
    splits = []

    for sample in samples_list:
        filename, sample_dict = sample

        assert len(sample_dict["features"]) == 1  # TODO multi feature
        for feature_name in sample_dict["features"]:
            if len(sample_dict["features"][feature_name]) - (context_left + context_right) > (
                    max_sequence_length + min_sequence_length) and max_sequence_length > 0:
                for i in range((len(sample_dict["features"][feature_name]) - (context_left + context_right)
                                + max_sequence_length - 1) // max_sequence_length):
                    if (len(sample_dict["features"][feature_name][
                            i * max_sequence_length + context_left:-context_right])
                            > max_sequence_length + min_sequence_length):
                        # we do not want to have sequences shorter than {min_sequence_length} but also do not want to discard sequences
                        # so we allow a few sequeces with length {max_sequence_length + min_sequence_length} instead
                        #####
                        # If the sequence length is above the threshold, we split it with a minimal length max/4
                        # If max length = 500, then the split will start at 500 + (500/4) = 625.
                        # A seq of length 625 will be splitted in one of 500 and one of 125
                        # filename_new = filename + "_c" + str(i)

                        total = len(sample_dict["features"][feature_name])

                        start_idx = context_left + i * max_sequence_length
                        end_idx = context_left + i * max_sequence_length + max_sequence_length
                        splits.append(
                            (filename, start_idx, end_idx))
                    else:
                        start_idx = context_left + i * max_sequence_length
                        end_idx = len(sample_dict["features"][feature_name]) - context_right
                        splits.append((filename, start_idx, end_idx))
                        break

            else:
                start_idx = context_left
                end_idx = len(sample_dict["features"][feature_name]) - context_right
                splits.append((filename, start_idx, end_idx))

    return splits


def get_order_by_length(feature_dict):
    ordering_length = {}
    for feature_name in feature_dict:
        ordering_length[feature_name] = \
            sorted(enumerate(feature_dict[feature_name]),
                   key=lambda _idx_filename: feature_dict[feature_name][_idx_filename[1]].shape[0])
        ordering_length[feature_name] = OrderedDict([(filename, {"idx": _idx,
                                                                 "length": feature_dict[feature_name][filename].shape[
                                                                     0]})
                                                     for _idx, filename in ordering_length[feature_name]])
    return ordering_length


def apply_context_single_feat(feat, context_left, context_right, start_idx, end_idx):
    _, num_feats = feat.shape
    length = end_idx - start_idx
    if isinstance(feat, np.ndarray):
        out_feat = \
            np.empty(
                (length,
                 num_feats,
                 context_left + context_right + 1)
            )
    elif isinstance(feat, torch.Tensor):
        out_feat = \
            torch.empty(
                (length,
                 num_feats,
                 context_left + context_right + 1), device=feat.device
            )
    else:
        raise ValueError
    for i in range(0, end_idx - start_idx):
        assert i + start_idx >= context_left, f"{i + start_idx} >= {context_left}"
        out_feat[i, :, :] = \
            feat[i + start_idx - context_left:i + start_idx + context_right + 1, :].transpose(1, 0)

    return out_feat
