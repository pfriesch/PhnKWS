import os
import json

import torch

from base import resume_checkpoint
from nn_.registries.model_registry import model_init
from utils.logger_config import logger
from utils.util import ensure_dir, folder_to_checkpoint


class BaseDecoder:
    """
    Base class for all decoders
    """

    def __init__(self, model_path):
        assert model_path.endswith(".pyt")
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

        if model_path:
            resume_checkpoint(model_path, self.model, logger)

    def is_keyword(self, input_features, keyword, sensitivity):
        output = self.model(input_features)
        # TODO
        return output
