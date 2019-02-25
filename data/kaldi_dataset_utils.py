from __future__ import print_function

import os
import os.path
import sys

import numpy as np
import torch

from data.data_util import load_features, splits_by_seqlen, filter_by_seqlen
from utils.logger_config import logger


def _filter_samples_by_length(file_names, feature_dict, features_loaded, label_dict, all_labels_loaded,
                              max_sample_len, min_sample_len):
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
            if type(max_sample_len) == int and \
                    len(features_loaded[feature_name][file]) > max_sample_len:
                logger.info("Skipping {}, feature of size {} too big ( {} expected) ".format(
                    file, len(features_loaded[feature_name][file]), max_sample_len))
                _continue = True
                break
            if type(min_sample_len) == int and \
                    min_sample_len > len(features_loaded[feature_name][file]):
                logger.info("Skipping {}, feature of size {} too small ( {} expected) ".format(
                    file, len(features_loaded[feature_name][file]), max_sample_len))
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


def _make_frames_shuffled(samples_list, main_feat, left_context, right_context):
    # framewise shuffled frames
    sample_splits = []
    # sample_splits = [ ( file_id, end_idx_total_in_chunk)]

    for sample_id, data in samples_list:
        sample_splits.extend([(sample_id, i)
                              for i in range(
                len(data['features'][main_feat]) - left_context - right_context)])

    return sample_splits


def _make_frames_sequential(samples_list, main_feat, aligned_labels, max_seq_len, left_context,
                            right_context):
    # sequential data
    if any([not aligned_labels[label_name] for label_name in aligned_labels]):
        assert all([not aligned_labels[label_name] for label_name in aligned_labels])
        # unaligned labels
        sample_splits, min_len = filter_by_seqlen(samples_list, max_seq_len,
                                                  left_context, right_context)
        logger.info(f"Used samples {len(sample_splits)}/{len(samples_list)} "
                    + f"for a max seq length of {max_seq_len} (min length was {min_len})")

    elif any([not aligned_labels[label_name] for label_name in aligned_labels]) \
            and not max_seq_len:
        assert all([not aligned_labels[label_name] for label_name in aligned_labels])
        # unaligned labels but no max_seq_len
        sample_splits = [
            (filename, left_context, len(sample_dict["features"][main_feat]) - right_context)
            for filename, sample_dict in samples_list]
    else:
        # framewise sequential
        if max_seq_len:
            sample_splits = splits_by_seqlen(samples_list, max_seq_len,
                                             left_context, right_context)
        else:
            raise NotImplementedError("Framewise without max_seq_len not impl")

    max_len = 0
    min_len = sys.maxsize

    for sample_id, start_idx, end_idx in sample_splits:
        max_len = (end_idx - start_idx) \
            if (end_idx - start_idx) > max_len else max_len

        min_len = (end_idx - start_idx) \
            if (end_idx - start_idx) < min_len else min_len

    # sort sigs/labels: longest -> shortest
    sample_splits = sorted(sample_splits, key=lambda x: x[2] - x[1])

    return sample_splits, max_len, min_len


def convert_chunk_from_kaldi_format(chnk_id_file_chnk, dataset_path, feature_dict, label_dict, all_labels_loaded,
                                    shuffle_frames, main_feat, aligned_labels, max_sample_len, min_sample_len,
                                    max_seq_len, left_context, right_context):
    chnk_id, file_chnk = chnk_id_file_chnk
    file_names = [feat.split(" ")[0] for feat in file_chnk]

    chnk_prefix = os.path.join(dataset_path, "chunk_{:04d}".format(chnk_id))

    features_loaded = {}
    for feature_name in feature_dict:
        chnk_scp = chnk_prefix + "feats.scp"
        with open(chnk_scp, "w") as f:
            f.writelines(file_chnk)

        features_loaded[feature_name] = load_features(chnk_scp, feature_dict[feature_name]["feature_opts"])
        os.remove(chnk_scp)

    samples = _filter_samples_by_length(file_names, feature_dict, features_loaded, label_dict,
                                        all_labels_loaded, max_sample_len, min_sample_len)

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

    max_len = None
    min_len = None
    if not shuffle_frames:
        sample_splits, max_len, min_len = _make_frames_sequential(samples_list, main_feat, aligned_labels, max_seq_len,
                                                                  left_context,
                                                                  right_context)

    else:
        sample_splits = _make_frames_shuffled(samples_list, main_feat, left_context, right_context)

    # samples =
    #   {file_id:
    #       { 'feautres' :
    #           { 'fbank' : ndarray
    #           },
    #         'labels':
    #           { 'lab_phn' : ndarray
    #           }
    #       }
    #   }

    # sample_splits = [ ( file_id, start_idx_file, end_idx_file)] # context & frame
    # sample_splits = [ ( file_id, end_idx_total_in_chunk)]

    torch.save(
        {"samples": samples,
         "means": mean,
         "std": std},
        chnk_prefix + ".pyt"
    )

    return chnk_id, [(chnk_id,) + sample_split for sample_split in sample_splits], max_len, min_len
    # TODO add warning when files get too big -> choose different chunk size
