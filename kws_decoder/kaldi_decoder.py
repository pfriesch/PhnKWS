import os
import json

import torch
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from base.base_trainer import KaldiOutputWriter
from base.utils import resume_checkpoint
from data.kaldi_dataset_utils import _load_labels
from kaldi_decoding_scripts.decode_dnn_custom_graph import decode
from kws_decoder.kalid_decoder_kw.prepare_decode_graph import make_kaldi_decoding_graph
from nn_.registries.model_registry import model_init
from utils.logger_config import logger
from utils.util import ensure_dir

import matplotlib.pyplot as plt

from utils.utils import feat_without_context


def plot(sample_name, input_feat, output, phn_dict, _labels=None, text=None):
    # def plot(sample_name, input_feat, output, phn_dict, decoded=None, _labels=None, text=None,
    #                                result_decoded=None):
    min_height = 0.10
    top_phns = [x[0] for x in list(sorted(enumerate(output.max(axis=0)), key=lambda x: x[1], reverse=True))
                if output[:, x[0]].max() > min_height]

    if _labels is not None:
        _labels = _labels['lab_mono'][sample_name]
        _labels = [phn_dict.idx2phoneme[l] for l in _labels]
        prev_phn = None
        _l_out = []
        _l_out_i = []

        for _i, l in enumerate(_labels):
            if prev_phn is None:
                prev_phn = l
                # _l_out.append("")
            else:
                if prev_phn == l:
                    pass
                # _l_out.append("")
                else:
                    _l_out.append(prev_phn)
                    _l_out_i.append(_i)
                    prev_phn = l

    if 0 in top_phns:
        top_phns.remove(0)  # TODO removed blank maybe add later

    # phn_dict = {k + 1: v for k, v in phn_dict.items()}
    # phn_dict[0] = "<blk>"
    # assert len(phn_dict) == output.shape[1]

    height = 500

    fig = plt.figure()
    ax = fig.subplots()
    # in_feat = feat_without_context(input_feat)
    in_feat = input_feat['fbank'].squeeze().numpy()
    ax.imshow(in_feat.T, origin='lower',
              # extent=[-(in_feat.shape[0] - output.shape[0] + 1) // 2, in_feat.shape[0], 0, 100],
              extent=[-(in_feat.shape[0] - output.shape[0]), in_feat.shape[0], 0, height],
              alpha=0.5)
    for i in top_phns:
        # ax.plot(output[:, i] * height, linewidth=0.5)
        if i != 0:
            # x = (output[:, i] * height).argmax()
            # y = (output[:, i] * height)[x]

            peaks, _ = find_peaks(output[:, i] * height, height=min_height * height, distance=10)
            # plt.plot(peaks, (output[:, i] * height)[peaks], "x", markersize=1)

            for peak in peaks:
                plt.axvline(x=peak, ymax=(output[:, i] * height)[peak] / height, linewidth=0.5, color='r',
                            linestyle='-')
                ax.annotate(phn_dict.reducedIdx2phoneme[i - 1], xy=(peak, (output[:, i] * height)[peak]), fontsize=4)
    # ax.
    if _labels is not None:
        ax.set_xticklabels(_l_out, rotation='vertical')
        ax.set_xticks(_l_out_i)
    # ax.legend()
    # ax.xaxis.set_major_locator(ticker.FixedLocator(_l_out_i))
    # ax.xaxis.set_(ticker.FixedLocator(_l_out_i))
    plt.tick_params(labelsize=4)
    ax.set_aspect(aspect=0.2)
    # if result_decoded is None:
    #     ax.set_title(result_decoded)
    fig.savefig(f"output_{sample_name}.png")
    fig.savefig(f"output_{sample_name}.pdf")
    fig.clf()
    # plt.close(fig)

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
        logger.info(self.model)
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

        plot_stuff = False
        # if plot_stuff:
        #     lab_dict = {"lab_mono": {
        #         "label_folder": "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/",
        #         "label_opts": "ali-to-phones --per-frame=true",
        #         "lab_data_folder": "/mnt/data/libs/kaldi/egs/librispeech/s5/data/dev_clean/",
        #         "lab_graph": "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b/graph_tgsmall/"
        #     }}
        #     label_index_from = 1
        #     _labels = _load_labels(lab_dict, label_index_from, max_label_length=None, phoneme_dict=self.phoneme_dict)

        with KaldiOutputWriter(tmp_out_dir, "keyword", self.model.out_names, self.epoch) as writer:
            output_label = 'out_cd'
            post_files.append(writer.post_file[output_label].name)
            for sample_name in tqdm(input_features, desc="computing acoustic features:", position=1):
                input_feature = {"fbank": self.preprocess_feat(input_features[sample_name])}
                # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
                input_feature["fbank"] = ((input_feature["fbank"] - mean) / std)
                if self.model.batch_ordering == "TNCL":
                    input_feature["fbank"] = input_feature["fbank"].permute(2, 0, 1).unsqueeze(3)
                _output = self.model(input_feature)

                assert output_label in _output
                output = _output[output_label]

                if self.model.batch_ordering == "NCL":
                    output = output.detach().squeeze(0).numpy()

                elif self.model.batch_ordering == "TNCL":
                    output = output.detach().squeeze(1).numpy().T

                # if self.config['test'][output_label]['normalize_posteriors']:
                # read the config file
                counts = self.config['dataset']['dataset_definition']['data_info']['labels']['lab_cd']['lab_count']
                if len(output) >= 3481:  # TODO make based on index from
                    output = output[1:] - np.log(counts / np.sum(counts)).reshape(-1, 1)
                else:
                    output = output - np.log(counts / np.sum(counts)).reshape(-1, 1)

                output = output.transpose()

                if plot_stuff:
                    # if plot_num < 5:
                    # output_exp = np.exp(_output['out_mono'].detach().squeeze(0).numpy())
                    # plot(sample_name, input_feature, output_exp, self.phoneme_dict.idx2phoneme,
                    #      _labels=_labels[sample_name])

                    output_exp = np.exp(_output['out_cd'].detach().squeeze(1).numpy())
                    plot(sample_name, input_feature, output_exp, self.phoneme_dict.idx2phoneme)
                    # plot_num += 1

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

    if context_right > 0:
        out_feat[:, context_left:-context_right:] = feat.transpose()
    else:
        out_feat[:, context_left:] = feat.transpose()
    return out_feat
