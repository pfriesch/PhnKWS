import torch

from data.data_util import make_big_chunk, load_kws, get_order_by_length
from utils.logger_config import logger
from utils.util import Timer
from utils.utils import make_timit_kws_labels


class SpeechCommandsDataset(object):

    def __init__(self, fea_dict,
                 label_dict,
                 kw2phn_mapping,
                 context_left,
                 context_right,
                 max_sequence_length,
                 tensorboard_logger,
                 debug=False,
                 local=False):

        self.tensorboard_logger = tensorboard_logger
        with Timer("init_dataset_elapsed_time_load", [self.tensorboard_logger, logger]) as t:
            assert local == False

            _feature_dict, _label_dict = load_kws(fea_dict, label_dict, kw2phn_mapping)

            if debug:
                for label_name in _label_dict:
                    _label_dict[label_name] = dict(
                        sorted(list(_label_dict[label_name].items()), key=lambda x: x[0])[:30])
                for feat_name in _feature_dict:
                    _feature_dict[feat_name] = dict(
                        sorted(list(_feature_dict[feat_name].items()), key=lambda x: x[0])[:30])

            # TODO handle context for ctc (padding) ?
            assert context_left == 0 and context_right == 0

            # TODO make multiple chunks if too big
            sample_name, feature_chunks, label_chunks = make_big_chunk(_feature_dict, _label_dict,
                                                                       label_start_zero=False)

            self.ordering_length = get_order_by_length(_feature_dict)

            self.feature_chunks = {feat_name: torch.from_numpy(feature_chunks[feat_name]).float()
                                   for feat_name, v in feature_chunks.items()}
            self.label_chunks = {label_name: torch.from_numpy(label_chunks[label_name]).long()
                                 for label_name, v in label_chunks.items()}
            self.sample_names, self.samples = zip(*list(sample_name.items()))

            self.feature_dim = {feat_name: v.shape[1] for feat_name, v in self.feature_chunks.items()}

    def move_to(self, device):
        with Timer("move_to_gpu_dataset_elapsed_time_load", [self.tensorboard_logger, logger]) as t:
            # Called "move to" to indicated difference to pyTorch .to(). This function mutates this object.
            self.feature_chunks = {k: v.to(device) for k, v in self.feature_chunks.items()}
            self.label_chunks = {k: v.to(device) for k, v in self.label_chunks.items()}

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


if __name__ == '__main__':
    kw2phn_mapping = make_timit_kws_labels()

    dataset = SpeechCommandsDataset(
        fea_dict={
            "fbank": {
                "feature_lst_path": "/mnt/data/libs/kaldi/egs/google_speech_commands/kws/data_kws/dev/feats.scp",
                "feature_opts": "apply-cmvn --utt2spk=ark:/mnt/data/libs/kaldi/egs/google_speech_commands/kws/data_kws/dev/utt2spk  ark:/mnt/data/libs/kaldi/egs/google_speech_commands/kws/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                "cw_left": 0,
                "cw_right": 0,
                "n_chunks": 1
            }
        }, label_dict={
            "lab_phn": {
                "text_file": "/mnt/data/libs/kaldi/egs/google_speech_commands/kws/data_kws/dev/text",
            }
        },
        kw2phn_mapping=kw2phn_mapping,
        context_left=0, context_right=0, max_sequence_length=1000, tensorboard_logger=None, debug=False, local=False)
    print("t")