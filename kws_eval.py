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


#
def test_kws(checkpoint_path, decode_experiment_name):
    data_folder = "/mnt/data/pytorch-kaldi/bench_data/speech_commands_v0.02"

    files, keywords = get_files_speech_commands(data_folder, "validation_list.txt")
    # _files, _ = get_files_speech_commands(data_folder, "testing_list.txt")
    # random.shuffle(files)
    # files = [f for f in files if "seven" in f]
    # files = files[:200]
    # files += _files
    if 'reduced' in decode_experiment_name:
        reduced_keywords = ['MARVIN',
                            'SHEILA',
                            'SEVEN',
                            'VISUAL',
                            'HAPPY',  # TODO remove bad performing keywords
                            'FOLLOW',
                            'LEARN']
        keywords = reduced_keywords

    keywords = {kw.upper(): _i for _i, kw in enumerate(keywords)}

    print(keywords)

    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    result_path = os.path.join(base_dir, f"result_kws_{decode_experiment_name}.json")
    print(f"result_path: {result_path}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(result_path):

        batch_size = 100
        # n_parallel = (len(files) // batch_size) + 1
        num_workers = 2 if 'DEBUG_MODE' not in os.environ or not os.environ['DEBUG_MODE'] else 1

        engine = KWSEngine(keywords, 0.0,
                           checkpoint_path)
        results = []

        file_chunks = [files[start_idx:start_idx + batch_size] for chunk_idx, start_idx in
                       enumerate(range(0, len(files), batch_size))]

        with tqdm(total=len(file_chunks), desc="total: ", position=0) as pbar:
            with Pool(num_workers) as p:
                for result in p.imap_unordered(engine.process_batch, file_chunks):
                    results.append(result)
                    pbar.update()
        # with tqdm(total=len(file_chunks), desc="total: ", position=0) as pbar:
        #     for file_chunk in file_chunks:
        #         result = engine.process_batch(file_chunk)
        #         results.append(result)

        _results = list(itertools.chain.from_iterable([r.items() for r in results]))
        _results = [(r[0], r[1][0], r[1][1], r[1][2], r[1][3]) for r in _results]
        with open(result_path, "w") as f:
            json.dump(_results, f)
    with open(result_path, "r") as f:
        results_loaded = json.load(f)
    # with open("result_dump.json", "r") as f:
    #     results = json.load(f)

    plot_result_confusion_matrix(keywords, results_loaded, f"{base_dir}/{decode_experiment_name}")


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
    # checkpoint_path = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ce/libri_WaveNetBIG_fbank_ce/checkpoints/checkpoint_e19.pth"
    # checkpoint_path = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ctc/libri_WaveNetBIG_fbank_ctc_PER_21percent/checkpoints/checkpoint_e37.pth"
    # checkpoint_path = "/mnt/data/pytorch-kaldi/trained_models/libri_WaveNetBig_ctc/libri_WaveNetBIG_fbank_ctc_PER26_from_scratch/checkpoints/checkpoint_e36_bias.pth"
    checkpoint_path = "/mnt/data/pytorch-kaldi/trained_models/libri_LSTM_fbank_ce/checkpoints/checkpoint_e0_gs316.pth"

    # test_kws(checkpoint_path, 'kws_reduced')
    # test_kws(checkpoint_path, 'kws_all')
    # test_asr()
    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    result_path = os.path.join(base_dir, "alexa_results/")
    snr_db = 994
    run_benchmark_alexa(result_path, checkpoint_path, snr_db)

    # run_benchmark_speech_commands(checkpoint_path, os.path.abspath("bench_output"))
