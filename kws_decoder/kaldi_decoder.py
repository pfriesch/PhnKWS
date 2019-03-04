import os
import json

import torch
import numpy as np
from tqdm import tqdm

from base.utils import resume_checkpoint
from kaldi_decoding_scripts.decode_dnn_custom_graph import decode
from kws_decoder.kalid_decoder_kw.prepare_decode_graph import make_kaldi_decoding_graph
from nn_.registries.model_registry import model_init
from trainer import KaldiOutputWriter
from utils.logger_config import logger
from utils.util import ensure_dir

import matplotlib.pyplot as plt


def plot(sample_name, output, phn_dict):
    top_phns = [x[0] for x in list(sorted(enumerate(output.max(axis=0)), key=lambda x: x[1], reverse=True))[:20]]

    fig = plt.figure()
    ax = fig.subplots()
    for i in top_phns:
        ax.plot(output[:, i], label=phn_dict[i])
    ax.legend()
    ax.set_title(sample_name)
    fig.savefig(f"output_{sample_name}.png")
    fig.clf()


class KaldiDecoder:
    def __init__(self, model_path, keywords, tmpdir):
        assert model_path.endswith(".pth")
        self.config = torch.load(model_path, map_location='cpu')['config']
        # TODO remove
        # self.config['exp']['save_dir'] = "/mnt/data/pytorch-kaldi/exp_TIMIT_MLP_FBANK"

        self.model = model_init(self.config)
        # TODO GPU decoding

        self.max_seq_length_train_curr = -1

        self.out_dir = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        # setup directory for checkpoint saving
        self.checkpoint_dir = os.path.join(self.out_dir, 'checkpoints')

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.out_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

        self.epoch, self.global_step, _ = resume_checkpoint(model_path, self.model, logger)

        self.phoneme_dict = self.config['dataset']['dataset_definition']['phoneme_dict']

        graph_dir = make_kaldi_decoding_graph(keywords, tmpdir)
        self.graph_path = os.path.join(graph_dir, "HCLG.fst")
        assert os.path.exists(self.graph_path)
        self.words_path = os.path.join(graph_dir, "words.txt")
        assert os.path.exists(self.words_path)
        self.alignment_model_path = os.path.join(graph_dir, "final.mdl")
        assert os.path.exists(self.alignment_model_path)

    def is_keyword_batch(self, input_features, sensitivity, tmp_out_dir=None):
        if tmp_out_dir is None:
            tmp_out_dir = self.out_dir

        all_samples_concat = None
        for sample_name, feat in tqdm(input_features.items()):
            if all_samples_concat is None:
                all_samples_concat = feat
            else:
                all_samples_concat = np.concatenate((all_samples_concat, feat))

        mean = torch.from_numpy(np.mean(all_samples_concat, axis=0)).to(dtype=torch.float32).unsqueeze(-1)
        std = torch.from_numpy(np.std(all_samples_concat, axis=0)).to(dtype=torch.float32).unsqueeze(-1)
        post_files = []

        post_files = []

        plot_num = 0

        with KaldiOutputWriter(tmp_out_dir, "keyword", self.model.out_names, self.epoch, self.config) as writer:
            output_label = 'out_cd'
            post_files.append(writer.post_file[output_label].name)
            for sample_name in tqdm(input_features, desc="computing acoustic features:"):
                input_feature = {"fbank": self.preprocess_feat(input_features[sample_name])}
                # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
                input_feature["fbank"] = ((input_feature["fbank"] - mean) / std)
                output = self.model(input_feature)
                assert output_label in output
                output = output[output_label]

                output = output.detach().squeeze(0).numpy()

                # if self.config['test'][output_label]['normalize_posteriors']:
                # read the config file
                counts = self.config['dataset']['dataset_definition']['data_info']['labels']['lab_cd']['lab_count']
                output = output[1:] - np.log(counts / np.sum(counts)).reshape(-1, 1)
                output = output.transpose()

                # if plot_num < 5:
                #     output_exp = np.exp(output)
                #     plot(sample_name, output_exp, self.phoneme_dict.idx2phoneme)
                #     plot_num += 1

                assert len(output.shape) == 2
                assert np.sum(np.isnan(output)) == 0, "NaN in output"
                writer.write_mat(output_label, output, sample_name)
        # self.config['decoding']['scoring_type'] = 'just_transcript'
        #### DECODING ####
        logger.debug("Decoding...")
        result = decode(**self.config['dataset']['dataset_definition']['decoding'],
                        alignment_model_path=self.alignment_model_path,
                        words_path=self.words_path,
                        graph_path=self.graph_path,
                        out_folder=tmp_out_dir,
                        featstrings=post_files)

        # TODO filter result

        return result

    def preprocess_feat(self, feat):
        assert len(feat.shape) == 2
        # length, num_feats = feat.shape
        feat_context = apply_context_single_feat(feat, self.model.context_left, self.model.context_right,
                                                 start_idx=self.model.context_left,
                                                 end_idx=len(feat) - self.model.context_right)

        return torch.from_numpy(feat_context).to(dtype=torch.float32).unsqueeze(0)


def apply_context_single_feat(feat, context_left, context_right, start_idx, end_idx, pad_context_zeros=True):
    _, num_feats = feat.shape
    # length = end_idx - start_idx
    # if length < 0:
    #     assert pad_context_zeros, "model has too big of a context so padding is necessary"

    if isinstance(feat, np.ndarray):
        out_feat = \
            np.zeros(
                (num_feats,
                 len(feat) + context_left + context_right)
            )
    elif isinstance(feat, torch.Tensor):
        out_feat = \
            torch.zeros(
                (num_feats,
                 len(feat) + context_left + context_right), device=feat.device
            )
    else:
        raise ValueError

    out_feat[:, context_left:-context_right:] = feat.transpose()
    return out_feat
