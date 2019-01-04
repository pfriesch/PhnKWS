import time

import numpy as np
import torch

import kaldi_io


def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


def read_lab_fea(fea_dict, lab_dict):
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

    # make sure labs and feats have the same features
    for fea in fea_loaded:
        fea_sample_ids = sorted(list(fea_loaded[fea].keys()))
        for lab in lab_loaded:
            lab_sample_ids = sorted(list(lab_loaded[lab].keys()))
            assert fea_sample_ids == lab_sample_ids

    return fea_loaded, lab_loaded


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

        # make sure the concat worked
        for filename in feat_dict[fea]:
            assert np.array_equal(feat_chunks[fea][sample_name[filename]['fea'][fea]['start_idx']:
                                                   sample_name[filename]['fea'][fea]['end_idx']],
                                  feat_dict[fea][filename])

    for lab in lab_dict:
        lab_chunks[lab] = np.concatenate(lab_chunks[lab])
        # make sure the concat worked
        for filename in lab_dict[lab]:
            assert np.array_equal(lab_chunks[lab][sample_name[filename]['lab'][lab]['start_idx']:
                                                  sample_name[filename]['lab'][lab]['end_idx']],
                                  lab_dict[lab][filename])

    if normalize_lab:
        for lab in lab_dict:
            lab_chunks[lab] = lab_chunks[lab] - lab_chunks[lab].min()

    if normalize_feat:
        for fea in feat_dict:
            feat_chunks[fea] = (feat_chunks[fea] - np.mean(feat_chunks[fea], axis=0)) / np.std(feat_chunks[fea], axis=0)

    return sample_name, feat_chunks, lab_chunks


class KaldiDataset(object):

    def __init__(self, fea_dict, lab_dict):
        feat_dict, lab_dict = read_lab_fea(fea_dict, lab_dict)

        # TODO split files that are too long
        # TODO handle context

        sample_name, feat_chunks, lab_chunks = make_big_chunk(feat_dict, lab_dict)
        self.feat_chunks = {fea: torch.from_numpy(feat_chunks[fea]).float()
                            for fea, v in feat_chunks.items()}
        self.lab_chunks = {lab: torch.from_numpy(lab_chunks[lab]).long()
                           for lab, v in lab_chunks.items()}
        self.sample_names, self.samples = zip(*list(sample_name.items()))

        self.fea_dim = {fea: v.shape[1] for fea, v in self.feat_chunks.items()}

    def move_to(self, device):
        # Called "move to" to indicated difference to pyTorch .to(). This function mutates this object.
        self.feat_chunks = {k: v.to(device) for k, v in self.feat_chunks.items()}
        self.lab_chunks = {k: v.to(device) for k, v in self.lab_chunks.items()}

    def __getitem__(self, index):
        return (self.sample_names[index],
                {fea: self.feat_chunks[fea][v['start_idx']:v['end_idx']]
                 for fea, v in self.samples[index]['fea'].items()},
                {lab: self.lab_chunks[lab][v['start_idx']:v['end_idx']]
                 for lab, v in self.samples[index]['lab'].items()})

    def __len__(self):
        return len(self.samples)
