import os
import json

import ctcdecode
import torch
import numpy as np
from tqdm import tqdm

from base.base_trainer import KaldiOutputWriter
from base.utils import resume_checkpoint
from data.kaldi_dataset_utils import _load_labels
from kaldi_decoding_scripts.ctc_decoding.decode_dnn_custom_graph import decode_ctc
from kws_decoder.eesen_decoder_kw.prepare_decode_graph import make_ctc_decoding_graph
from nn_.registries.model_registry import model_init
from utils.logger_config import logger
from utils.util import ensure_dir
from utils.utils import plot_alignment_spectrogram


class CTCDecoder:
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

        graph_dir = make_ctc_decoding_graph(keywords, self.phoneme_dict.phoneme2reducedIdx, tmpdir,
                                            draw_G_L_fsts=True)
        self.graph_path = os.path.join(graph_dir, "TLG.fst")
        assert os.path.exists(self.graph_path)
        self.words_path = os.path.join(graph_dir, "words.txt")
        # self.alignment_model_path = os.path.join(graph_dir, "final.mdl")
        # assert os.path.exists(self.alignment_model_path)

    # def test_decoder(self, phns="SIL S EH V AH N SIL"):
    #     _phn_idx = [self.phoneme_dict.phoneme2reducedIdx[p] + 1 for p in phns.split(" ")]
    #     test_output = np.ones((88, 42))
    #     _p = 0
    #     for i in range(88):
    #         if i % len(_phn_idx) == 6 and _p < len(_phn_idx):
    #             test_output[i][_phn_idx[_p]] = 1000000
    #             _p += 1
    #         else:
    #             test_output[i][0] = 1000000
    #
    #     test_output = (test_output.T / np.sum(test_output, axis=1)).T
    #
    #     test_output = np.log(test_output)
    #
    #     return test_output

    def is_keyword_batch(self, input_features, sensitivity, tmp_out_dir=None):
        if tmp_out_dir is None:
            tmp_out_dir = self.out_dir

        # https://stackoverflow.com/questions/15638612/calculating-mean-and-standard-deviation-of-the-data-which-does-not-fit-in-memory
        #
        # _, feat = next(iter(input_features.items()))
        # _dim = feat.shape[-1]
        #
        # n = 0
        # mean = np.zeros((_dim))
        # M2 = np.zeros((_dim))
        #
        # for sample_name, feat in tqdm(input_features.items()):
        #     # for i in range(10):
        #     for i in range(feat.shape[0]):
        #         n += 1
        #         delta = feat[i, :] - mean
        #         mean = mean + (delta / n)
        #         M2 = M2 + (delta ** 2)
        #
        # std = np.sqrt(M2 / (n - 1))
        # mean = torch.from_numpy(mean).to(dtype=torch.float32).unsqueeze(-1)
        # std = torch.from_numpy(std).to(dtype=torch.float32).unsqueeze(-1)

        # test_output = self.test_decoder()

        # plot_phns = metadata_dict is None
        plot_phns = False
        # if plot_phns:
        #     lab_dict = {"lab_mono": {
        #         "label_folder": "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/",
        #         "label_opts": "ali-to-phones --per-frame=true",
        #         "lab_data_folder": "/mnt/data/libs/kaldi/egs/librispeech/s5/data/dev_clean/",
        #         "lab_graph": "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b/graph_tgsmall/"
        #     }}
        #     label_index_from = 1
        #     _labels = _load_labels(lab_dict, label_index_from, max_label_length=None, phoneme_dict=self.phoneme_dict)
        #
        #     lab_dict = {"lab_mono": {
        #         "label_folder": "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/",
        #         "label_opts": "ali-to-phones",
        #         "lab_data_folder": "/mnt/data/libs/kaldi/egs/librispeech/s5/data/dev_clean/",
        #         "lab_graph": "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b/graph_tgsmall/"
        #     }}
        #     label_index_from = 1
        #     _labels_no_ali = _load_labels(lab_dict, label_index_from, max_label_length=None,
        #                                   phoneme_dict=self.phoneme_dict)

        vocabulary_size = 42
        vocabulary = [chr(c) for c in list(range(65, 65 + 58)) + list(range(65 + 58 + 69, 65 + 58 + 69 + 500))][
                     :vocabulary_size]
        decoder = ctcdecode.CTCBeamDecoder(vocabulary, log_probs_input=True, beam_width=1)

        all_samples_concat = None
        for sample_name, feat in tqdm(input_features.items()):
            if all_samples_concat is None:
                all_samples_concat = feat
            else:
                all_samples_concat = np.concatenate((all_samples_concat, feat))

        mean = torch.from_numpy(np.mean(all_samples_concat, axis=0)).to(dtype=torch.float32).unsqueeze(-1)
        std = torch.from_numpy(np.std(all_samples_concat, axis=0)).to(dtype=torch.float32).unsqueeze(-1)
        post_files = []

        plot_num = 0

        # len = 88

        # input_batch = []
        # sample_names = []
        # for sample_name in tqdm(input_features, desc="computing acoustic features:"):
        #     input_feature = self.preprocess_feat(input_features[sample_name])
        #     # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
        #     _input_feature = ((input_feature - mean) / std).unsqueeze(1)
        #     if _input_feature.shape[0] < len:
        #         _zeros = torch.zeros((88, 1, 40, 11))
        #         _zeros[-_input_feature.shape[0]:, :, :, :] = _input_feature
        #         _input_feature = _zeros
        #     input_batch.append(_input_feature)
        #     sample_names.append(sample_name)

        # input_batch = {'fbank': torch.cat(input_batch, dim=1)}

        beam_results = {}
        output_label = 'out_phn'
        assert output_label in self.model.out_names
        with KaldiOutputWriter(tmp_out_dir, "keyword", [output_label], self.epoch) as writer:
            post_files.append(writer.post_file[output_label].name)
            for sample_name in tqdm(input_features, desc="computing acoustic features:", position=1):
                # input_feature = {"fbank": self.preprocess_feat(input_features[sample_name])}
                input_feature = {"fbank": torch.from_numpy(input_features[sample_name].T).unsqueeze(0)}
                # Normalize over whole chunk instead of only over a single file, which is done by applying the kaldi cmvn
                input_feature["fbank"] = ((input_feature["fbank"] - mean) / std)

                # assert input_feature["fbank"].shape[2] > self.model.context_left + self.model.context_right + 50
                if input_feature["fbank"].shape[2] < self.model.context_left + self.model.context_right + 100:
                    padd = torch.zeros((input_feature["fbank"].shape[0], input_feature["fbank"].shape[1],
                                        self.model.context_left + self.model.context_right),
                                       device=input_feature["fbank"].device,
                                       dtype=input_feature["fbank"].dtype)
                    input_feature["fbank"] = torch.cat((padd, input_feature["fbank"]), dim=2)

                output = self.model(input_feature)
                assert output_label in output
                output = output[output_label]

                _logits = output.detach().permute(0, 2, 1)

                output = output.detach().squeeze(0).numpy().T
                # output = test_output

                # if self.config['test'][output_label]['normalize_posteriors']:
                # counts = self.config['dataset']['dataset_definition']['data_info']['labels']['lab_phn']['lab_count']
                # counts = np.array(counts)
                # blank_count = sum(counts)  # heuristic sil * 2 for the moment
                # counts = counts * 0.5
                # counts = np.concatenate((np.array([np.e]), counts))
                # blank_scale = 1.0
                # TODO try different blank_scales 4.0 5.0 6.0 7.0
                # counts[0] /= blank_scale
                # for i in range(1, 8):
                #     counts[i] /= noise_scale #TODO try noise_scale for SIL SPN etc I guess

                # prior = counts / np.sum(counts)

                # output[:, 1:] = output[:, 1:] - np.log(prior)
                # assert _logits.shape[0] == batch_size
                # output = np.exp(output)

                beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(_logits)
                beam_result = beam_result[0, 0, :out_seq_len[0, 0]]
                result_decoded = [self.phoneme_dict.reducedIdx2phoneme[l.item() - 1] for l in beam_result]
                result_decoded = " ".join(result_decoded)

                beam_results[sample_name] = result_decoded

                if plot_num < 20 and plot_phns:
                    # logger.debug(sample_name)

                    # logger.debug(result_decoded)
                    # if plot_phns:
                    #     label_decoded = " ".join(
                    #         [self.phoneme_dict.idx2phoneme[l.item()] for l in _labels_no_ali['lab_mono'][sample_name]])
                    #     logger.debug(label_decoded)

                    # if plot_phns:
                    #     plot_alignment_spectrogram(sample_name, input_feature["fbank"],
                    #                                (np.exp(output).T / np.exp(output).sum(axis=1)).T,
                    #                                self.phoneme_dict, _labels, result_decoded=result_decoded)
                    # else:
                    plot_alignment_spectrogram(sample_name, input_feature["fbank"],
                                               (np.exp(output).T / np.exp(output).sum(axis=1)).T,
                                               self.phoneme_dict, result_decoded=result_decoded)

                    plot_num += 1
                # else:
                #     beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(_logits)
                #     beam_result = beam_result[0, 0, :out_seq_len[0, 0]]
                #     # logger.debug(sample_name)
                #     result_decoded = [self.phoneme_dict.reducedIdx2phoneme[l.item() - 1] for l in beam_result]
                #     result_decoded = " ".join(result_decoded)
                #     # logger.debug(result_decoded)
                #     plot_alignment_spectrogram(sample_name, input_feature["fbank"],
                #                                (np.exp(output).T / np.exp(output).sum(axis=1)).T,
                #                                self.phoneme_dict, metadata_dict[sample_name], result_decoded=result_decoded)
                #
                #     plot_num += 1

                assert len(output.shape) == 2
                assert np.sum(np.isnan(output)) == 0, "NaN in output"
                assert output.shape[1] == len(self.phoneme_dict.reducedIdx2phoneme) + 1
                writer.write_mat(output_label, output.squeeze(), sample_name)

        # self.config['decoding']['scoring_type'] = 'just_transcript'
        #### DECODING ####
        logger.debug("Decoding...")
        result = decode_ctc(**self.config['dataset']['dataset_definition']['decoding'],
                            words_path=self.words_path,
                            graph_path=self.graph_path,
                            out_folder=tmp_out_dir,
                            featstrings=post_files)

        # TODO filter result

        return result

    # def preprocess_feat(self, feat):
    #     assert len(feat.shape) == 2
    #     length, num_feats = feat.shape
    #     assert length > self.model.context_left + self.model.context_right
    #     feat_context = apply_context_single_feat(feat, self.model.context_left, self.model.context_right,
    #                                              start_idx=self.model.context_left,
    #                                              end_idx=len(feat) - self.model.context_right)
    #
    #     return torch.from_numpy(feat_context).to(dtype=torch.float32)
