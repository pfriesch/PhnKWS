from __future__ import print_function

import os
import os.path
import sys

import numpy as np
import torch

from data.data_util import load_features, splits_by_seqlen, filter_by_seqlen, load_labels
from utils.logger_config import logger


def _load_labels(label_dict, label_index_from, max_label_length, phoneme_dict):
    all_labels_loaded = {}

    for lable_name in label_dict:
        all_labels_loaded[lable_name] = load_labels(label_dict[lable_name]['label_folder'],
                                                    label_dict[lable_name]['label_opts'])

        if max_label_length is not None and max_label_length > 0:
            all_labels_loaded[lable_name] = \
                {l: all_labels_loaded[lable_name][l] for l in all_labels_loaded[lable_name]
                 if len(all_labels_loaded[lable_name][l]) < max_label_length}

        if lable_name == "lab_phn":
            if phoneme_dict is not None:
                for sample_id in all_labels_loaded[lable_name]:
                    assert max(all_labels_loaded[lable_name][sample_id]) <= max(
                        phoneme_dict.idx2reducedIdx.keys()), \
                        "Are you sure you have the righ phoneme dict?" + \
                        " Labels have higher indices than phonemes ( {} <!= {} )".format(
                            max(all_labels_loaded[lable_name][sample_id]),
                            max(phoneme_dict.idx2reducedIdx.keys()))

                    # map labels according to phoneme dict
                    tmp_labels = np.copy(all_labels_loaded[lable_name][sample_id])
                    for k, v in phoneme_dict.idx2reducedIdx.items():
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
    if not aligned_labels:
        # unaligned labels
        sample_splits, min_len = filter_by_seqlen(samples_list, max_seq_len,
                                                  left_context, right_context)
        logger.info(f"Used samples {len(sample_splits)}/{len(samples_list)} "
                    + f"for a max seq length of {max_seq_len} (min length was {min_len})")

    elif not aligned_labels and not max_seq_len:
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
