import torch

from data.data_util import load_data, apply_context, make_big_chunk_no_order
from utils.logger_config import logger
from utils.util import Timer


class KaldiDatasetFramewiseShuffledFrames(object):

    def __init__(self, feature_dict,
                 label_dict,
                 context_left,
                 context_right,
                 max_sequence_length,
                 tensorboard_logger):
        self.tensorboard_logger = tensorboard_logger
        with Timer("init_dataset_elapsed_time_load", [self.tensorboard_logger, logger]) as t:
            _feature_dict, _label_dict = load_data(feature_dict, label_dict, max_sequence_length,
                                                   context_left + context_right)

            _feature_dict, _label_dict = apply_context(_feature_dict, _label_dict, context_left, context_right)

            # TODO make multiple chunks if too big
            feature_chunks, label_chunks = make_big_chunk_no_order(_feature_dict, _label_dict)

            self.feature_chunks = {feat_name: torch.from_numpy(feature_chunks[feat_name]).float()
                                   for feat_name, v in feature_chunks.items()}
            self.label_chunks = {label_name: torch.from_numpy(label_chunks[label_name]).long()
                                 for label_name, v in label_chunks.items()}

            self.feature_dim = {feat_name: v.shape[1] for feat_name, v in self.feature_chunks.items()}

    def move_to(self, device):
        with Timer("move_to_gpu_dataset_elapsed_time_load", [self.tensorboard_logger, logger]) as t:
            # Called "move to" to indicated difference to pyTorch .to(). This function mutates this object.
            self.feature_chunks = {k: v.to(device) for k, v in self.feature_chunks.items()}
            self.label_chunks = {k: v.to(device) for k, v in self.label_chunks.items()}

    def __getitem__(self, index):
        return ({feat_name: self.feature_chunks[feat_name][index]
                 for feat_name in self.feature_chunks},
                {lab_name: self.label_chunks[lab_name][index]
                 for lab_name in self.label_chunks})

    def __len__(self):
        return len(self.feature_chunks[next(iter(self.feature_chunks))])

    def save(self, path):
        torch.save({
            "ordering_length": self.ordering_length,
            "feature_chunks": self.feature_chunks,
            "label_chunks": self.label_chunks,
            "sample_names": self.sample_names,
            "samples": self.samples,
            "feature_dim": self.feature_dim}, path)

    def load(self, path):
        _load_dict = torch.load(path)
        self.ordering_length = _load_dict["ordering_length"]
        self.feature_chunks = _load_dict["feature_chunks"]
        self.label_chunks = _load_dict["label_chunks"]
        self.sample_names = _load_dict["sample_names"]
        self.samples = _load_dict["samples"]
        self.feature_dim = _load_dict["feature_dim"]
