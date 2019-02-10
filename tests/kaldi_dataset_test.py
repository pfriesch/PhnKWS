import os
import random

from data.kaldi_dataset import KaldiDataset
from data.phoneme_dicts.phoneme_dict import get_phoneme_dict
from utils.logger_config import logger

KALDI_ROOT = os.environ["KALDI_ROOT"]
assert os.path.exists(f"{KALDI_ROOT}/egs"), f"{KALDI_ROOT}/egs not found!"


# TODO add tests that test the edge cases of chunks -> index

def test_unaligned(tmpdir):
    logger.configure_logger(out_folder=tmpdir)
    dataset = KaldiDataset(cache_data_root=tmpdir,
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": f"{KALDI_ROOT}/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": f"apply-cmvn --utt2spk=ark:{KALDI_ROOT}/egs/timit/s5/data/train/utt2spk  ark:{KALDI_ROOT}/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones",
                                   "lab_count_file": "none",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph",
                               }
                           },
                           max_sample_len=1000,
                           left_context=10, right_context=2,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict(f"{KALDI_ROOT}/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=False

                           )

    filename, features, lables = next(iter(dataset))
    for feature_name in features:
        feat_len = features[feature_name].shape[0]
        assert list(features[feature_name].shape[1:]) == [40, 13]
    for label_name in lables:
        assert lables[label_name].shape[0] < feat_len


def test_force_aligned_sequential(tmpdir):
    logger.configure_logger(out_folder=tmpdir)

    dataset = KaldiDataset(cache_data_root=tmpdir,
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": f"{KALDI_ROOT}/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": f"apply-cmvn --utt2spk=ark:{KALDI_ROOT}/egs/timit/s5/data/train/utt2spk  ark:{KALDI_ROOT}/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_cd_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-pdf",
                                   "lab_count_file": "auto",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               },
                               "lab_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones --per-frame=true",
                                   "lab_count_file": "none",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               }
                           },
                           max_sample_len=1000,
                           left_context=10, right_context=2,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict(f"{KALDI_ROOT}/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=100

                           )
    filename, features, lables = next(iter(dataset))
    for feature_name in features:
        feat_len = features[feature_name].shape[0]
        assert list(features[feature_name].shape[1:]) == [40, 13]
    for label_name in lables:
        assert lables[label_name].shape[0] == feat_len


def test_force_aligned_shuffle_frames(tmpdir):
    logger.configure_logger(out_folder=tmpdir)

    dataset = KaldiDataset(cache_data_root=tmpdir,
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": f"{KALDI_ROOT}/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": f"apply-cmvn --utt2spk=ark:{KALDI_ROOT}/egs/timit/s5/data/train/utt2spk  ark:{KALDI_ROOT}/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_cd_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-pdf",
                                   "lab_count_file": "auto",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               },
                               "lab_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones --per-frame=true",
                                   "lab_count_file": "none",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               }
                           },
                           max_sample_len=None,
                           left_context=10, right_context=2,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict(f"{KALDI_ROOT}/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=False,
                           shuffle_frames=True

                           )
    for i in range(1000):
        filename, features, lables = dataset[random.randint(0, len(dataset))]
        for feature_name in features:
            assert list(features[feature_name].shape) == [1, 40, 13]
        for label_name in lables:
            assert list(lables[label_name].shape) == [1]


def test_unaligned_no_context(tmpdir):
    logger.configure_logger(out_folder=tmpdir)

    dataset = KaldiDataset(cache_data_root=tmpdir,
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": f"{KALDI_ROOT}/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": f"apply-cmvn --utt2spk=ark:{KALDI_ROOT}/egs/timit/s5/data/train/utt2spk  ark:{KALDI_ROOT}/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones",
                                   "lab_count_file": "none",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph",
                               }
                           },
                           max_sample_len=1000,
                           left_context=0, right_context=0,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict(f"{KALDI_ROOT}/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=False

                           )
    filename, features, lables = next(iter(dataset))
    for feature_name in features:
        feat_len = features[feature_name].shape[0]
        assert list(features[feature_name].shape[1:]) == [40, 1]
    for label_name in lables:
        assert lables[label_name].shape[0] < feat_len


def test_force_aligned_sequential_no_context(tmpdir):
    logger.configure_logger(out_folder=tmpdir)

    dataset = KaldiDataset(cache_data_root=tmpdir,
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": f"{KALDI_ROOT}/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": f"apply-cmvn --utt2spk=ark:{KALDI_ROOT}/egs/timit/s5/data/train/utt2spk  ark:{KALDI_ROOT}/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_cd_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-pdf",
                                   "lab_count_file": "auto",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               },
                               "lab_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones --per-frame=true",
                                   "lab_count_file": "none",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               }
                           },
                           max_sample_len=1000,
                           left_context=0, right_context=0,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict(f"{KALDI_ROOT}/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=100

                           )
    filename, features, lables = next(iter(dataset))
    for feature_name in features:
        feat_len = features[feature_name].shape[0]
        assert list(features[feature_name].shape[1:]) == [40, 1]
    for label_name in lables:
        assert lables[label_name].shape[0] == feat_len


def test_force_aligned_shuffle_frames_no_context(tmpdir):
    logger.configure_logger(out_folder=tmpdir)

    dataset = KaldiDataset(cache_data_root=tmpdir,
                           dataset_name="TIMIT_tr",
                           feature_dict={
                               "fbank": {
                                   "feature_lst_path": f"{KALDI_ROOT}/egs/timit/s5/data/train/feats_fbank.scp",
                                   "feature_opts": f"apply-cmvn --utt2spk=ark:{KALDI_ROOT}/egs/timit/s5/data/train/utt2spk  ark:{KALDI_ROOT}/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |",
                               }
                           },
                           label_dict={
                               "lab_cd_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-pdf",
                                   "lab_count_file": "auto",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               },
                               "lab_phn": {
                                   "label_folder": f"{KALDI_ROOT}/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
                                   "label_opts": "ali-to-phones --per-frame=true",
                                   "lab_count_file": "none",
                                   "lab_data_folder": f"{KALDI_ROOT}/egs/timit/s5/data/train/",
                                   "lab_graph": f"{KALDI_ROOT}/egs/timit/s5/exp/tri3/graph"
                               }
                           },
                           max_sample_len=None,
                           left_context=0, right_context=0,
                           normalize_features=True,
                           phoneme_dict=get_phoneme_dict(f"{KALDI_ROOT}/egs/timit/s5/data/lang/phones.txt",
                                                         stress_marks=True, word_position_dependency=False),
                           split_files_max_seq_len=False,
                           shuffle_frames=True

                           )
    for i in range(1000):
        filename, features, lables = dataset[random.randint(0, len(dataset))]
        for feature_name in features:
            assert list(features[feature_name].shape) == [1, 40, 1]
        for label_name in lables:
            assert list(lables[label_name].shape) == [1]
