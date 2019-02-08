import os
import json

import torch

from base.utils import resume_checkpoint
from data.data_util import apply_context_single_feat, load_counts
from nn_.registries.model_registry import model_init
from utils.logger_config import logger
from utils.util import ensure_dir, folder_to_checkpoint
import numpy as np


class BaseDecoder:
    """
    Base class for all decoders
    """

    def __init__(self, model_path):
        assert model_path.endswith(".pth")
        self.config = torch.load(model_path, map_location='cpu')['config']

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

        resume_checkpoint(model_path, self.model, logger)

    def is_keyword(self, input_features, keyword, sensitivity):
        output = self.model(input_features)
        output_label = 'out_cd'
        assert output_label in output
        output = output[output_label]

        output = output.detach().numpy()

        if self.config['test'][output_label]['normalize_posteriors']:
            # read the config file
            counts = load_counts(
                self.config['test'][output_label]['normalize_with_counts_from_file'])
            output = output - np.log(counts / np.sum(counts))

        output = np.exp(output)
        return output

    def preprocess_feat(self, feat):
        assert len(feat.shape) == 2
        # length, num_feats = feat.shape
        feat_context = apply_context_single_feat(feat, self.model.context_left, self.model.context_right)

        return torch.from_numpy(feat_context).to(dtype=torch.float32).unsqueeze(1)
