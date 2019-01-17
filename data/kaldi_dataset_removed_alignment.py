import time

import torch

from data.data_util import load_data, apply_context, make_big_chunk, get_order_by_length, collapse_alignment


class KaldiDatasetRemovedAlignment(object):

    def __init__(self, feature_dict, label_dict, context_left, context_right, max_sequence_length, tensorboard_logger,
                 debug=False, local=False):
        start_time = time.time()
        assert local == False

        _feature_dict, _label_dict = load_data(feature_dict, label_dict, max_sequence_length,
                                               context_left + context_right)

        if debug:
            for label_name in _label_dict:
                _label_dict[label_name] = dict(
                    sorted(list(_label_dict[label_name].items()), key=lambda x: x[0])[:30])
            for feat_name in _feature_dict:
                _feature_dict[feat_name] = dict(
                    sorted(list(_feature_dict[feat_name].items()), key=lambda x: x[0])[:30])

        # TODO split files that are too long
        _label_dict = apply_context(_label_dict, context_left, context_right)

        #TODO collaps alignment
        _label_dict = collapse_alignment(_label_dict)

        # TODO make multiple chunks if too big
        sample_name, feature_chunks, label_chunks = make_big_chunk(_feature_dict, _label_dict)

        self.ordering_length = get_order_by_length(_feature_dict)

        self.feature_chunks = {feat_name: torch.from_numpy(feature_chunks[feat_name]).float()
                               for feat_name, v in feature_chunks.items()}
        self.label_chunks = {label_name: torch.from_numpy(label_chunks[label_name]).long()
                             for label_name, v in label_chunks.items()}
        self.sample_names, self.samples = zip(*list(sample_name.items()))

        self.feature_dim = {feat_name: v.shape[1] for feat_name, v in self.feature_chunks.items()}

        elapsed_time_load = time.time() - start_time
        self.tensorboard_logger = tensorboard_logger
        self.tensorboard_logger.add_scalar("init_dataset", elapsed_time_load)

    def move_to(self, device):
        start_time = time.time()

        # Called "move to" to indicated difference to pyTorch .to(). This function mutates this object.
        self.feature_chunks = {k: v.to(device) for k, v in self.feature_chunks.items()}
        self.label_chunks = {k: v.to(device) for k, v in self.label_chunks.items()}

        elapsed_time_load = time.time() - start_time
        self.tensorboard_logger.add_scalar("move_to_gpu_dataset", elapsed_time_load)

    def _get_by_filename(self, filename):
        index = self.sample_names.index(filename)
        return ({feat_name: self.feature_chunks[feat_name][v['start_idx']:v['end_idx']]
                 for feat_name, v in self.samples[index]['features'].items()},
                {lab_name: self.label_chunks[lab_name][v['start_idx']:v['end_idx']]
                 for lab_name, v in self.samples[index]['labels'].items()})

    def __getitem__(self, index):
        return (self.sample_names[index],
                {feat_name: self.feature_chunks[feat_name][v['start_idx']:v['end_idx']]
                 for feat_name, v in self.samples[index]['features'].items()},
                {lab_name: self.label_chunks[lab_name][v['start_idx']:v['end_idx']]
                 for lab_name, v in self.samples[index]['labels'].items()})

    def __len__(self):
        return len(self.samples)

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
