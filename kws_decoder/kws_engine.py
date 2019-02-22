import os
import tempfile

from base.base_decoder import get_decoder
from data import kaldi_io
from utils.logger_config import logger
from utils.utils import run_shell, check_environment
from ww_benchmark.engine import Engine


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

    def __init__(self, keywords, sensitivity, model_path) -> None:
        super().__init__()
        self.tmp_dir = tempfile.TemporaryDirectory()
        # TODO debug mode

        logger.configure_logger(self.tmp_dir.name)
        check_environment()

        assert isinstance(keywords, list)
        self.keywords = keywords
        self.sensitivity = sensitivity

        self.decoder = get_decoder(model_path, keywords, self.tmp_dir.name)

    def process_batch(self, wav_files):
        _wav_files = set()
        for file in wav_files:
            assert os.path.abspath(file)
            if file in _wav_files:
                logger.warn(f"Duplicate file {file}, ignoring...")
            else:
                _wav_files.add(file)
        wav_files = list(_wav_files)

        tmp_scp, spk2utt_path, utt2spk_path = self.preppare_tmp_files(wav_files, self.tmp_dir.name)
        feats = get_kaldi_feats(tmp_scp, self.tmp_dir.name, spk2utt_path, utt2spk_path)

        #TODO apply mean std norm, same as in dataloader

        result = self.decoder.is_keyword_batch(feats, self.sensitivity)
        return result

    def release(self):
        self.tmp_dir.cleanup()

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
        # TODO add noise for context of model
        files = [(os.path.basename(file).split("_")[0],
                  "_".join(file.rsplit("/", 2)[1:])
                  [:-4], file) for file in files]

        tmp_scp = os.path.join(tmp_dir, "tmp.scp")
        with open(tmp_scp, "w") as f:
            f.writelines([f"{file_id} {path}\n" for speaker, file_id, path in files])

        #### spk2utt
        spk2utt_path = os.path.join(tmp_dir, "spk2utt")
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
        utt2spk_path = os.path.join(tmp_dir, "utt2spk")
        utt2spk = {}
        for speaker, file_id, path in files:
            utt2spk[file_id] = speaker
        with open(utt2spk_path, "w") as f:
            f.writelines([f"{file_id} {speaker}\n" for file_id, speaker in utt2spk.items()])
        #### /utt2spk
        return tmp_scp, spk2utt_path, utt2spk_path

# def plot_output_phonemes(model_logits):
#     for filename, logits in model_logits.items():
#         #### P1
#
#         # just_max_val = logits.max(axis=2)[:, 0]
#         # fig, axs = plt.subplots(1, 1)
#         # axs.plot(just_max_val)
#         # fig.tight_layout()
#         # plt.savefig("just_max_val.png")
#
#         #### P2
#
#         max_20 = sorted(logits.argmax(axis=2).squeeze(), reverse=True)[:20]
#         log_max_20 = logits[:, :, max_20]
#
#         fig, axs = plt.subplots(20, 1)
#         for i in range(20):
#             axs[i].plot(logits[:, :, i].squeeze())
#         # fig.tight_layout()
#         plt.savefig("max_20.png")
