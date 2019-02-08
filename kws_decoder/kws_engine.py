import os
import torch
from os.path import join as join_path

from base.base_decoder import BaseDecoder
from data import kaldi_io
from utils.logger_config import logger
from utils.utils import run_shell, check_environment
from ww_benchmark.engine import Engine
import numpy as np

import matplotlib.pyplot as plt


# TODO kaldi feat extraction for a single file
# cmvn from training data
# get fst to decode lattices


# TODO instead of relying on proounciation lexicon

# run asr and plot the log likelyhood of the individual phonemes over time

def get_kaldi_feats(scp_file, out_dir, spk2utt, utt2spk):
    # Compute features
    fbank_config = "kaldi_decoding_scripts/conf/fbank.conf"
    compress = "true"
    name = "decoding"
    assert os.path.exists(fbank_config)
    out_scp = f"{out_dir}/raw_fbank_{name}.scp"
    out_ark = f"{out_dir}/raw_fbank_{name}.ark"
    run_shell(f"compute-fbank-feats --verbose=2 --config={fbank_config} scp,p:{scp_file} ark:- | \
    copy-feats --compress={compress} ark:- ark,scp:{out_ark},{out_scp}")

    # Compute normalization data
    cmvn_ark = f"{out_dir}/cmvn_{name}.ark"
    cmvn_scp = f"{out_dir}/cmvn_{name}.scp"
    run_shell(f"compute-cmvn-stats --spk2utt=ark:{spk2utt} scp:{out_scp} ark,scp:{cmvn_ark},{cmvn_scp}")

    # Load normalized
    feature_opts = f"apply-cmvn --utt2spk=ark:{utt2spk} ark:{cmvn_ark} ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |"
    features_loaded = \
        {k: m for k, m in
         kaldi_io.read_mat_ark(f'ark:copy-feats scp:{out_scp} ark:- | {feature_opts}')}

    return features_loaded


class KWSEngine(Engine):

    def __init__(self, keyword, sensitivity, model_path) -> None:
        super().__init__()
        self.tmp_dir = "/mnt/data/pytorch-kaldi/tmp"
        logger.configure_logger(self.tmp_dir)
        check_environment()

        self.keyword = keyword
        self.sensitivity = sensitivity

        self.decoder = BaseDecoder(model_path)

    def process(self, wav_files):
        for file in wav_files:
            assert os.path.abspath(file)

        tmp_scp, spk2utt_path, utt2spk_path = self.preppare_tmp_files(wav_files, self.tmp_dir)
        feats = get_kaldi_feats(tmp_scp, self.tmp_dir, spk2utt_path, utt2spk_path)

        return {filename:
                    self.decoder.is_keyword({"fbank": self.decoder.preprocess_feat(feat)},
                                            self.keyword, self.sensitivity)
                for filename, feat in feats.items()}

    def release(self):
        pass

    def __str__(self):
        pass

    @property
    def frame_length(self):
        return -1  # TODO

    @staticmethod
    def sensitivity_range(engine_type):
        return super().sensitivity_range(engine_type)

    @staticmethod
    def create(engine_type, keyword, sensitivity):
        return super().create(engine_type, keyword, sensitivity)

    @staticmethod
    def preppare_tmp_files(files, tmp_dir):
        # [(speaker, file_id, path]
        files = [(os.path.basename(file).split("_")[0], os.path.basename(file)[:-4], file) for file in files]

        tmp_scp = join_path(tmp_dir, "tmp.scp")
        with open(tmp_scp, "w") as f:
            f.writelines([f"{file_id} {path}\n" for speaker, file_id, path in files])

        #### spk2utt
        spk2utt_path = join_path(tmp_dir, "spk2utt")
        spk2utt = {}
        for speaker, file_id, path in files:
            if speaker in spk2utt:
                spk2utt[speaker].append(file_id)
            else:
                spk2utt[speaker] = [file_id]
        with open(spk2utt_path, "w") as f:
            f.writelines([f"{speaker} {' '.join(file_ids)}\n" for speaker, file_ids in spk2utt.items()])
        #### /spk2utt

        #### utt2spk
        utt2spk_path = join_path(tmp_dir, "utt2spk")
        utt2spk = {}
        for speaker, file_id, path in files:
            utt2spk[file_id] = speaker
        with open(utt2spk_path, "w") as f:
            f.writelines([f"{file_id} {speaker}\n" for file_id, speaker in utt2spk.items()])
        #### /utt2spk
        return tmp_scp, spk2utt_path, utt2spk_path


def plot_output_phonemes(model_logits):
    for filename, logits in model_logits.items():
        #### P1

        # just_max_val = logits.max(axis=2)[:, 0]
        # fig, axs = plt.subplots(1, 1)
        # axs.plot(just_max_val)
        # fig.tight_layout()
        # plt.savefig("just_max_val.png")

        #### P2

        max_20 = sorted(logits.argmax(axis=2).squeeze(), reverse=True)[:20]
        log_max_20 = logits[:, :, max_20]

        fig, axs = plt.subplots(20, 1)
        for i in range(20):
            axs[i].plot(logits[:, :, i].squeeze())
        # fig.tight_layout()
        plt.savefig("max_20.png")


def test():
    engine = KWSEngine("", 0.0,
                       "/mnt/data/pytorch-kaldi/exp/libri_TDNN_fbank_20190205_025557/checkpoints/checkpoint-epoch9.pth")

    data_folder = "/mnt/data/libs/kaldi/egs/google_speech_commands/kws/data_kws/speech_commands_v0.02"

    files = [join_path(data_folder, "bed/20a0d54b_nohash_0.wav"),
             join_path(data_folder, "bed/1ed557b9_nohash_0.wav")]
    model_logits = engine.process(files)
    plot_output_phonemes(model_logits)


if __name__ == '__main__':
    test()
