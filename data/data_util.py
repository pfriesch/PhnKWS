import os
from collections import OrderedDict, Counter, namedtuple

import numpy as np

from data import kaldi_io
from utils.logger_config import logger


def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


# def load_data_unaligned(feature_dict, label_dict, max_sequence_length):
#     features_loaded = {}
#     labels_loaded = {}
#
#     for feature_name in feature_dict:
#         feature_lst_path = feature_dict[feature_name]['feature_lst_path']
#         feature_opts = feature_dict[feature_name]['feature_opts']
#
#         num_ignored = 0
#
#         def short_enough(_m):
#             nonlocal num_ignored
#             _short_enough = max_sequence_length < 0 or len(_m) < max_sequence_length
#             num_ignored += int(not _short_enough)
#             return _short_enough
#
#         features_loaded[feature_name] = \
#             {k: m for k, m in
#              kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(feature_lst_path, feature_opts))
#              if short_enough(m)}
#         logger.info("Ignored {} features in data loading since they where too long".format(num_ignored))
#         assert len(features_loaded[feature_name]) > 0
#
#     phn_mapping = raw_mapping
#     for label_name in label_dict:
#         # TODO remove starting space
#         phn_transcripts = json.load(open(label_dict[label_name]['phn_transcripts']))
#         labels_loaded[label_name] = {t['id']: np.array([phn_mapping[p] for p in t['phn']])
#                                      for t in phn_transcripts if
#                                      t['id'] in features_loaded[feature_name]}
#         # total_phn_count = json.load(open(label_dict[label_name]['total_phn_count']))
#
#     return features_loaded, labels_loaded


def load_data_unaligned(feature_dict, label_dict, phn_mapping, max_sequence_length):
    features_loaded = {}
    labels_loaded = {}

    for feature_name in feature_dict:
        feature_lst_path = feature_dict[feature_name]['feature_lst_path']
        feature_opts = feature_dict[feature_name]['feature_opts']

        num_ignored = 0

        def short_enough(_m):
            nonlocal num_ignored
            _short_enough = max_sequence_length < 0 or len(_m) < max_sequence_length
            num_ignored += int(not _short_enough)
            return _short_enough

        features_loaded[feature_name] = \
            {k: m for k, m in
             kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(feature_lst_path, feature_opts))
             if short_enough(m)}
        logger.info("Ignored {} features in data loading since they where too long".format(num_ignored))
        assert len(features_loaded[feature_name]) > 0

    for label_name in label_dict:

        label_folder = label_dict[label_name]['label_folder']
        label_opts = label_dict[label_name]['label_opts']

        _phn_mapping = phn_mapping[label_name]
        debug_removed_ids = Counter()

        def map_label(label):
            nonlocal _phn_mapping
            #### Debugging
            for _lab_id in label:
                if _lab_id not in _phn_mapping.id_mapping:
                    debug_removed_ids[_lab_id] += 1
            #### /Debugging

            labels_new = [_phn_mapping.id_mapping[_lab_id] for _lab_id in label if _lab_id in _phn_mapping.id_mapping]

            if 0 in labels_new:
                print("shit")

            # We probably do not want to remove repeating phonemes since we do not know if there is a silence between them. Also it can't hurt too much to detect the same phoneneme twice?!...
            ## Remove repeating characters
            ## labels_new = [i for i, _ in itertools.groupby(labels_new)]
            return np.array(labels_new)

        for feature_name in feature_dict:
            # Note that I'm copying only the aligments of the loaded feature
            labels_loaded[label_name] = \
                {k: map_label(v) for k, v in
                 kaldi_io.read_vec_int_ark(
                     'gunzip -c {}/ali*.gz | {} {}/final.mdl ark:- ark:-|'
                         .format(label_folder, label_opts, label_folder))
                 if k in features_loaded[feature_name]}

            logger.info("removed these indices: {}".format(
                {next(t[0] for t in _phn_mapping.all_phone_info if _id == t[1]): count
                 for _id, count in debug_removed_ids.items()}))

            # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")
            features_loaded[feature_name] = \
                {k: v for k, v in list(features_loaded[feature_name].items()) if
                 k in labels_loaded[label_name]}

            # check length of label_name and feat
            for filename in features_loaded[feature_name]:
                if not features_loaded[feature_name][filename].shape[0] > 0:
                    logger.warn("file {} has 0 length feature".format(filename))
                if not len(labels_loaded[label_name][filename]) > 0:
                    logger.warn("file {} has 0 length label".format(filename))

    return features_loaded, labels_loaded


def load_features(feature_dict):
    features_loaded = {}

    for feature_name in feature_dict:
        feature_lst_path = feature_dict[feature_name]['feature_lst_path']
        feature_opts = feature_dict[feature_name]['feature_opts']

        features_loaded[feature_name] = \
            {k: m for k, m in
             kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(feature_lst_path, feature_opts))}
        assert len(features_loaded[feature_name]) > 0

    return features_loaded


def load_data(feature_dict, label_dict, max_sequence_length, context_size):
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

        label_folder = label_dict[label_name]['label_folder']
        label_opts = label_dict[label_name]['label_opts']

        for feature_name in feature_dict:
            # Note that I'm copying only the aligments of the loaded feature
            labels_loaded[label_name] = \
                {k: v for k, v in
                 kaldi_io.read_vec_int_ark(
                     'gunzip -c {}/ali*.gz | {} {}/final.mdl ark:- ark:-|'
                         .format(label_folder, label_opts, label_folder))
                 if k in features_loaded[feature_name]}
            # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")
            features_loaded[feature_name] = \
                {k: v for k, v in list(features_loaded[feature_name].items()) if
                 k in labels_loaded[label_name]}

            # check length of label_name and feat matching
            for filename in features_loaded[feature_name]:
                assert features_loaded[feature_name][filename].shape[0] == len(labels_loaded[label_name][filename])

    # TODO remove with 1/4 of max length -> add to config
    # TODO add option weather the context_size is applied to the minimum sequence length
    min_sequence_length = max_sequence_length // 4 + context_size

    features_loaded_chunked = {k: {} for k in features_loaded}
    labels_loaded_chunked = {k: {} for k in labels_loaded}

    for feature_name in features_loaded:
        for filename in sorted(features_loaded[feature_name],
                               key=lambda _filename: len(features_loaded[feature_name][_filename])):
            if len(features_loaded[feature_name][filename]) > max_sequence_length and max_sequence_length > 0:
                for i in range((len(
                        features_loaded[feature_name][filename]) + max_sequence_length - 1) // max_sequence_length):
                    if (len(features_loaded[feature_name][filename][i * max_sequence_length:])
                            > max_sequence_length + min_sequence_length):
                        # we do not want to have sequences shorter than {min_sequence_length} but also do not want to discard sequences
                        # so we allow a few sequeces with length {max_sequence_length + min_sequence_length} instead
                        #####
                        # If the sequence length is above the threshold, we split it with a minimal length max/4
                        # If max length = 500, then the split will start at 500 + (500/4) = 625.
                        # A seq of length 625 will be splitted in one of 500 and one of 125

                        filename_new = filename + "_c" + str(i)

                        features_loaded_chunked[feature_name][filename_new] = features_loaded[feature_name][filename][
                                                                              i * max_sequence_length:
                                                                              i * max_sequence_length + max_sequence_length]
                        for label_name in labels_loaded:
                            labels_loaded_chunked[label_name][filename_new] = labels_loaded[label_name][filename][
                                                                              i * max_sequence_length:
                                                                              i * max_sequence_length + max_sequence_length]
                    else:
                        filename_new = filename + "_c" + str(i)

                        features_loaded_chunked[feature_name][filename_new] = features_loaded[feature_name][filename][
                                                                              i * max_sequence_length:]
                        for label_name in labels_loaded:
                            labels_loaded_chunked[label_name][filename_new] = labels_loaded[label_name][filename][
                                                                              i * max_sequence_length:]

                        break

            else:
                features_loaded_chunked[feature_name][filename] = features_loaded[feature_name][filename]
                for label_name in labels_loaded:
                    labels_loaded_chunked[label_name][filename] = labels_loaded[label_name][filename]

    return features_loaded_chunked, labels_loaded_chunked


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


def apply_context(label_dict, context_left, context_right):
    for label_name in label_dict:
        for filename in label_dict[label_name]:
            label_dict[label_name][filename] = label_dict[label_name][filename][
                                               context_left:len(label_dict[label_name][filename]) - context_right]
    return label_dict


def make_big_chunk(feature_dict, label_dict, normalize_feat=True, label_start_zero=True):
    sample_name = {k: {'features': {}, 'labels': {}} for k in feature_dict[list(feature_dict.keys())[0]].keys()}
    feature_chunks = {}
    label_chunks = {}

    for feature_name in feature_dict:
        feature_chunks[feature_name] = []
        idx = 0
        for filename in feature_dict[feature_name]:
            sample_name[filename]['features'][feature_name] = {}
            sample = feature_dict[feature_name][filename]
            sample_name[filename]['features'][feature_name]['len'] = sample.shape[0]
            sample_name[filename]['features'][feature_name]['start_idx'] = idx
            sample_name[filename]['features'][feature_name]['end_idx'] = idx + sample.shape[0]
            feature_chunks[feature_name].append(sample)
            idx = idx + sample.shape[0]

    if label_dict is not None:
        for label_name in label_dict:
            label_chunks[label_name] = []
            idx = 0
            for filename in label_dict[label_name]:
                sample_name[filename]['labels'][label_name] = {}
                sample = label_dict[label_name][filename]
                sample_name[filename]['labels'][label_name]['len'] = sample.shape[0]
                sample_name[filename]['labels'][label_name]['start_idx'] = idx
                sample_name[filename]['labels'][label_name]['end_idx'] = idx + sample.shape[0]
                label_chunks[label_name].append(sample)
                idx = idx + sample.shape[0]

    for feature_name in feature_dict:
        feature_chunks[feature_name] = np.concatenate(feature_chunks[feature_name])

        # # make sure the concat worked
        # for filename in feat_dict[fea]:
        #     assert np.array_equal(feature_chunks[fea][sample_name[filename]['features'][fea]['start_idx']:
        #                                            sample_name[filename]['features'][fea]['end_idx']],
        #                           feat_dict[fea][filename])

    if label_dict is not None:
        for label_name in label_dict:
            label_chunks[label_name] = np.concatenate(label_chunks[label_name])
            # # make sure the concat worked
            # for filename in lab_dict[lab]:
            #     assert np.array_equal(label_chunks[lab][sample_name[filename]['labels'][lab]['start_idx']:
            #                                           sample_name[filename]['labels'][lab]['end_idx']],
            #                           lab_dict[lab][filename])

    if label_dict is not None:
        if label_start_zero:
            for label_name in label_dict:
                label_chunks[label_name] = label_chunks[label_name] - label_chunks[label_name].min()
        else:
            for label_name in label_dict:
                # No zero label e.g. for ctc loss with blank label at 0
                if label_chunks[label_name].min() == 0:
                    label_chunks[label_name] = label_chunks[label_name] + 1

    if normalize_feat:
        for feature_name in feature_dict:
            feature_chunks[feature_name] = (feature_chunks[feature_name] - np.mean(feature_chunks[feature_name],
                                                                                   axis=0)) / np.std(
                feature_chunks[feature_name], axis=0)

    return sample_name, feature_chunks, label_chunks

