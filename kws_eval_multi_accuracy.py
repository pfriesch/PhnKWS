from multiprocessing.pool import Pool

from utils.logger_config import logger
from utils.utils import plot_result_confusion_matrix

import json
import os
import random
from glob import glob
from tqdm import tqdm

from kws_decoder.kws_engine import KWSEngine
import numpy as np
import itertools
import matplotlib.pyplot as plt

from ww_benchmark.benchmark import run_benchmark_alexa
from ww_benchmark.benchmark import run_benchmark_speech_commands


def get_files_speech_commands(data_folder, file_list):
    with open(os.path.join(data_folder, file_list), "r")as f:
        file_list = f.readlines()

    keywords = set()
    files = []
    for file in file_list:
        file = file.strip()
        keywords.add(os.path.dirname(file))
        files.append(os.path.join(data_folder, file))

    return files, list(keywords)


def get_files_librispeech(data_folder):
    files = []
    for file in glob(os.path.join(data_folder, "*", "*", "*.flac")):
        file = file.strip()
        files.append(os.path.join(data_folder, file))

    return files


def gett_acc(keywords, results):
    keywords = list(keywords.keys())
    #### confusion_matrix
    confusion_matrix = np.zeros((len(keywords) + 1, len(keywords) + 1))
    keywords = ["<UNK>"] + keywords
    for sample_id, transcript, lattice_confidence, lm_posterior, acoustic_posterior in results:
        transcript = transcript[0]
        gt = sample_id.split("_", 1)[0].upper()
        if gt not in keywords:
            gt = "<UNK>"

        gt_index = keywords.index(gt)
        transcript_index = keywords.index(transcript)
        confusion_matrix[gt_index, transcript_index] += 1
        # # print(transcript, gt)

    #### count

    corret_prediction = (confusion_matrix * np.eye(confusion_matrix.shape[0])).sum(axis=1)
    total = confusion_matrix.sum(axis=1)
    accuracy = (corret_prediction / total)[1:]

    # sorting = [_i for _i, acc in sorted(enumerate(accuracy), key=lambda x: x[1], reverse=True)]
    # accuracy = [accuracy[_i] for _i in sorting]
    return list(zip(keywords[1:], accuracy))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # b1 = ax.bar(np.arange(0, len(count_gt), dtype=float) - 0.125, count_gt, width=0.25, align='center')
    # b2 = ax.bar(np.arange(0, len(accuracy), dtype=float), accuracy, align='center')
    # plt.xticks(tick_marks, [keywords[1:][_i] for _i in sorting], rotation=90, fontsize=8)
    #
    # # ax.legend((b1, b2), ('# Ground Truth', '# Predicted'))
    # plt.savefig(f"{out_path}/accuracy.png")
    # plt.savefig(f"{out_path}/accuracy.pdf")
    # plt.clf()

    # print(json.dumps(results, indent=1))
    # plot_output_phonemes(model_logits)


#
def test_kws_multi_acc(checkpoint_path_ctc, checkpoint_path_ce, checkpoint_path_lstm, decode_experiment_name):
    data_folder = "/mnt/data/pytorch-kaldi/bench_data/speech_commands_v0.02"

    _, keywords = get_files_speech_commands(data_folder, "validation_list.txt")

    keywords = {kw.upper(): _i for _i, kw in enumerate(keywords)}

    print(keywords)
    sort_id = 2

    resutls_list = []

    for checkpoint_path in [checkpoint_path_ctc, checkpoint_path_ce, checkpoint_path_lstm]:
        base_dir = os.path.dirname(os.path.dirname(checkpoint_path))

        result_path = os.path.join(base_dir, f"result_kws_{decode_experiment_name}.json")
        print(f"result_path: {result_path}")

        assert os.path.exists(result_path)
        with open(result_path, "r") as f:
            results_loaded = json.load(f)

        resutls_list.append(gett_acc(keywords, results_loaded))

    print("t")

    sorting = [_i for _i, acc in sorted(enumerate(resutls_list[sort_id]), key=lambda x: x[1][1], reverse=True)]
    # accuracy = [accuracy[_i] for _i in sorting]

    keywords = [resutls_list[sort_id][_i][0] for _i in sorting]

    tick_marks = np.arange(len(keywords))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = ['WaveNet CTC', 'WaveNet CE', 'LSTM']
    pos = [-0.3, 0, 0.3]
    for i, accuracies in enumerate(resutls_list):
        accuracies = [accuracies[_i] for _i in sorting]
        print(accuracies)
        # b1 = ax.bar(np.arange(0, len(count_gt), dtype=float) - 0.125, count_gt, width=0.25, align='center')
        width = 0.3
        b2 = ax.bar(np.arange(0, len(accuracies), dtype=float) + pos[i], [a[1] for a in accuracies],
                    width=width, align='center', label=labels[i])

    ax.legend()

    plt.xticks(tick_marks, keywords, rotation=90, fontsize=8)

    # ax.legend((b1, b2), ('# Ground Truth', '# Predicted'))
    plt.savefig(f"accuracy_multi_sorted_lstm.png")
    plt.savefig(f"accuracy_multi_sorted_lstm.pdf")
    plt.clf()


#
#
# def test_asr():
#     data_folder = "/mnt/data/datasets/LibriSpeech/dev-clean"
#
#     data_folder_kw = "/mnt/data/pytorch-kaldi/bench_data/speech_commands_v0.02"
#
#     _, keywords = get_files_speech_commands(data_folder_kw, "validation_list.txt")
#
#     files = get_files_librispeech(data_folder)
#     files = files[:40]
#     keywords = [kw.upper() for kw in keywords]
#     # print(keywords)
#
#     engine = KWSEngine(keywords, 0.0,
#                        "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ctc/libri_WaveNetBIG_fbank_ctc_PER26_from_scratch/checkpoints/checkpoint_e36.pth",
#                        n_parallel=1)
#
#     results = engine.process_batch(files)
#     plot_result_confusion_matrix(keywords, results)


if __name__ == '__main__':
    checkpoint_path_ctc = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ctc/libri_WaveNetBIG_fbank_ctc_PER_21percent/checkpoints/checkpoint_e37.pth"
    # checkpoint_path = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ctc/libri_WaveNetBIG_fbank_ctc_PER26_from_scratch/checkpoints/checkpoint_e36_bias.pth"

    # checkpoint_path = "/mnt/data/pytorch-kaldi/exp/libri_WaveNetBIG_fbank_ctc/checkpoints/checkpoint_e10.pth"
    checkpoint_path_ce = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ce/libri_WaveNetBIG_fbank_ce/checkpoints/checkpoint_e19.pth"
    # checkpoint_path = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBIG_fbank_ctc/checkpoints/checkpoint_e8.pth"
    checkpoint_path_lstm = "/mnt/data/pytorch-kaldi/trained_models/libri_LSTM_fbank_ce/checkpoints/checkpoint_e0_gs316.pth"

    # test_kws(checkpoint_path, 'kws_reduced')
    test_kws_multi_acc(checkpoint_path_ctc, checkpoint_path_ce, checkpoint_path_lstm, 'kws_all')
    # test_asr()

    # base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    # result_path = os.path.join(base_dir, "alexa_results/")
    # snr_db = 994
    # run_benchmark_alexa(result_path, checkpoint_path, snr_db)

    # run_benchmark_speech_commands(checkpoint_path, os.path.abspath("bench_output"))
