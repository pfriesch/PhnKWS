from collections import OrderedDict

import numpy as np

from data_loader import kaldi_io


def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


def read_fea(fea_dict):
    fea_loaded = {}

    for fea in fea_dict:
        fea_scp = fea_dict[fea]['fea_lst']
        fea_opts = fea_dict[fea]['fea_opts']

        fea_loaded[fea] = {k: m for k, m in
                           kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(fea_scp, fea_opts))}
        assert len(fea_loaded[fea]) > 0

    return fea_loaded


def read_lab_fea(fea_dict, lab_dict, max_sequence_length, context_size):
    fea_loaded = {}
    lab_loaded = {}

    for fea in fea_dict:
        fea_scp = fea_dict[fea]['fea_lst']
        fea_opts = fea_dict[fea]['fea_opts']

        fea_loaded[fea] = {k: m for k, m in
                           kaldi_io.read_mat_ark('ark:copy-feats scp:{} ark:- |{}'.format(fea_scp, fea_opts))}
        assert len(fea_loaded[fea]) > 0

    for lab in lab_dict:

        lab_folder = lab_dict[lab]['lab_folder']
        lab_opts = lab_dict[lab]['lab_opts']

        for fea in fea_dict:
            # Note that I'm copying only the aligments of the loaded fea
            lab_loaded[lab] = {k: v for k, v in kaldi_io.read_vec_int_ark(
                'gunzip -c {}/ali*.gz | {} {}/final.mdl ark:- ark:-|'.format(lab_folder, lab_opts, lab_folder))
                               if k in fea_loaded[fea]}
            # This way I remove all the features without an aligment (see log file in alidir "Did not Succeded")
            fea_loaded[fea] = {k: v for k, v in list(fea_loaded[fea].items()) if k in lab_loaded[lab]}

            # check length of lab and feat matching
            for filename in fea_loaded[fea]:
                assert fea_loaded[fea][filename].shape[0] == len(lab_loaded[lab][filename])

    # TODO remove with 1/4 of max length -> add to config
    # TODO add option weather the context_size is applied to the minimum sequence length
    min_sequence_length = max_sequence_length // 4 + context_size

    fea_loaded_chunked = {k: {} for k in fea_loaded}
    lab_loaded_chunked = {k: {} for k in lab_loaded}

    for fea in fea_loaded:
        for filename in fea_loaded[fea]:
            if len(fea_loaded[fea][filename]) > max_sequence_length and max_sequence_length > 0:
                for i in range((len(fea_loaded[fea][filename]) + max_sequence_length - 1) // max_sequence_length):
                    if (len(fea_loaded[fea][filename][i * max_sequence_length:])
                            > max_sequence_length + min_sequence_length):
                        # we do not want to have sequences shorter than {min_sequence_length} but also do not want to discard sequences
                        # so we allow a few sequeces with length {max_sequence_length + min_sequence_length} instead
                        #####
                        # If the sequence length is above the threshold, we split it with a minimal length max/4
                        # If max length = 500, then the split will start at 500 + (500/4) = 625.
                        # A seq of length 625 will be splitted in one of 500 and one of 125

                        filename_new = filename + "_c" + str(i)

                        fea_loaded_chunked[fea][filename_new] = fea_loaded[fea][filename][i * max_sequence_length:
                                                                                          i * max_sequence_length + max_sequence_length]
                        for lab in lab_loaded:
                            lab_loaded_chunked[lab][filename_new] = lab_loaded[lab][filename][i * max_sequence_length:
                                                                                              i * max_sequence_length + max_sequence_length]
                    else:
                        filename_new = filename + "_c" + str(i)

                        fea_loaded_chunked[fea][filename_new] = fea_loaded[fea][filename][i * max_sequence_length:]
                        for lab in lab_loaded:
                            lab_loaded_chunked[lab][filename_new] = lab_loaded[lab][filename][i * max_sequence_length:]

                        break

    else:
        fea_loaded_chunked[fea][filename] = fea_loaded[fea][filename]
        for lab in lab_loaded:
            lab_loaded_chunked[lab][filename] = lab_loaded[lab][filename]

    return fea_loaded_chunked, lab_loaded_chunked


def get_order_by_length(feat_dict):
    ordering_length = {}
    for fea in feat_dict:
        ordering_length[fea] = \
            sorted(enumerate(feat_dict[fea]),
                   key=lambda _idx_filename: feat_dict[fea][_idx_filename[1]].shape[0])
        ordering_length[fea] = OrderedDict([(filename, {"idx": _idx,
                                                        "length": feat_dict[fea][filename].shape[0]})
                                            for _idx, filename in ordering_length[fea]])
    return ordering_length


def apply_context(lab_dict, context_left, context_right):
    for lab in lab_dict:
        for filename in lab_dict[lab]:
            lab_dict[lab][filename] = lab_dict[lab][filename][context_left:len(lab_dict[lab][filename]) - context_right]
    return lab_dict


def make_big_chunk(feat_dict, lab_dict, normalize_feat=True, normalize_lab=True):
    sample_name = {k: {'fea': {}, 'lab': {}} for k in feat_dict[list(feat_dict.keys())[0]].keys()}
    feat_chunks = {}
    lab_chunks = {}

    for fea in feat_dict:
        feat_chunks[fea] = []
        idx = 0
        for filename in feat_dict[fea]:
            sample_name[filename]['fea'][fea] = {}
            sample = feat_dict[fea][filename]
            sample_name[filename]['fea'][fea]['len'] = sample.shape[0]
            sample_name[filename]['fea'][fea]['start_idx'] = idx
            sample_name[filename]['fea'][fea]['end_idx'] = idx + sample.shape[0]
            feat_chunks[fea].append(sample)
            idx = idx + sample.shape[0]

    if lab_dict is not None:
        for lab in lab_dict:
            lab_chunks[lab] = []
            idx = 0
            for filename in lab_dict[lab]:
                sample_name[filename]['lab'][lab] = {}
                sample = lab_dict[lab][filename]
                sample_name[filename]['lab'][lab]['len'] = sample.shape[0]
                sample_name[filename]['lab'][lab]['start_idx'] = idx
                sample_name[filename]['lab'][lab]['end_idx'] = idx + sample.shape[0]
                lab_chunks[lab].append(sample)
                idx = idx + sample.shape[0]

    for fea in feat_dict:
        feat_chunks[fea] = np.concatenate(feat_chunks[fea])

        # # make sure the concat worked
        # for filename in feat_dict[fea]:
        #     assert np.array_equal(feat_chunks[fea][sample_name[filename]['fea'][fea]['start_idx']:
        #                                            sample_name[filename]['fea'][fea]['end_idx']],
        #                           feat_dict[fea][filename])

    if lab_dict is not None:
        for lab in lab_dict:
            lab_chunks[lab] = np.concatenate(lab_chunks[lab])
            # # make sure the concat worked
            # for filename in lab_dict[lab]:
            #     assert np.array_equal(lab_chunks[lab][sample_name[filename]['lab'][lab]['start_idx']:
            #                                           sample_name[filename]['lab'][lab]['end_idx']],
            #                           lab_dict[lab][filename])

    if lab_dict is not None:
        if normalize_lab:
            for lab in lab_dict:
                lab_chunks[lab] = lab_chunks[lab] - lab_chunks[lab].min()

    if normalize_feat:
        for fea in feat_dict:
            feat_chunks[fea] = (feat_chunks[fea] - np.mean(feat_chunks[fea], axis=0)) / np.std(feat_chunks[fea], axis=0)

    return sample_name, feat_chunks, lab_chunks
