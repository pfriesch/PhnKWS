import os
import json

import torch
import numpy as np
from tqdm import tqdm

from base.utils import resume_checkpoint
from data.data_util import apply_context_single_feat
from kaldi_decoding_scripts.decode_dnn_custom_graph import decode
from kws_decoder.kalid_decoder.prepare_decode_graph import make_kaldi_decoding_graph
from nn_.registries.model_registry import model_init
from trainer import KaldiOutputWriter
from utils.logger_config import logger
from utils.util import ensure_dir


class BaseDecoder:
    """
    Base class for all decoders
    """

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

        self.epoch, self.global_step, self.decoding_norm_data = resume_checkpoint(model_path, self.model, logger)

        graph_dir = make_kaldi_decoding_graph(keywords, tmpdir)
        self.graph_path = os.path.join(graph_dir, "HCLG.fst")
        self.words_path = os.path.join(graph_dir, "words.txt")
        self.alignment_model_path = os.path.join(graph_dir, "final.mdl")

    def is_keyword_batch(self, input_features, sensitivity):
        post_files = []

        with KaldiOutputWriter(self.out_dir, "keyword", self.epoch, self.config) as writer:
            output_label = 'out_cd'
            post_files.append(writer.post_file[output_label].name)
            for sample_name in tqdm(input_features, desc="computing acoustic features:"):
                input_feature = {"fbank": self.preprocess_feat(input_features[sample_name])}
                output = self.model(input_feature)
                assert output_label in output
                output = output[output_label]

                output = output.detach().squeeze(1).numpy()

                if self.config['test'][output_label]['normalize_posteriors']:
                    # read the config file
                    counts = self.decoding_norm_data[output_label]["normalize_with_counts"]
                    output = output - np.log(counts / np.sum(counts))

                output = np.exp(output)

                assert len(output.shape) == 2
                writer.write_mat(output_label, output.squeeze(), sample_name)
        self.config['decoding']['scoring_type'] = 'just_transcript'
        #### DECODING ####
        logger.debug("Decoding...")
        result = decode(**self.config['decoding'],
                        alignment_model_path=self.alignment_model_path,
                        words_path=self.words_path,
                        graph_path=self.graph_path,
                        out_folder=self.out_dir,
                        featstrings=post_files)

        #TODO filter result

        return result

    def preprocess_feat(self, feat):
        assert len(feat.shape) == 2
        # length, num_feats = feat.shape
        feat_context = apply_context_single_feat(feat, self.model.context_left, self.model.context_right,
                                                 start_idx=self.model.context_left,
                                                 end_idx=len(feat) - self.model.context_right)

        return torch.from_numpy(feat_context).to(dtype=torch.float32).unsqueeze(1)
