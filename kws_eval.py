from multiprocessing.pool import Pool

from utils.utils import plot_result_confusion_matrix

import json
import os
import random
from glob import glob
from tqdm import tqdm

from kws_decoder.kws_engine import KWSEngine
import numpy as np
import itertools

from ww_benchmark.benchmark import run_benchmark


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


def test_kws():
    data_folder = "/mnt/data/pytorch-kaldi/bench_data/speech_commands_v0.02"

    files, keywords = get_files_speech_commands(data_folder, "validation_list.txt")
    # random.shuffle(files)
    # files = [f for f in files if "seven" in f]
    # files = files[:200]
    keywords = [kw.upper() for kw in keywords]
    keywords = sorted(keywords)

    print(keywords)

    batch_size = 100
    # n_parallel = (len(files) // batch_size) + 1
    n_parallel = os.cpu_count()

    engine = KWSEngine(keywords, 0.0,
                       # "/mnt/data/pytorch-kaldi/exp_finished_runs_backup/libri_MLP_fbank_20190225_133944/checkpoints/checkpoint-epoch7.pth",
                       "/mnt/data/pytorch-kaldi/exp_finished_runs_backup/libri_WaveNet_fbank_framewise/checkpoints/checkpoint-epoch5.pth",
                       n_parallel)
    results = []

    chunks = list()
    file_chunks = [(files[start_idx:start_idx + batch_size], chunk_idx)
                   for chunk_idx, start_idx in enumerate(range(0, len(files), batch_size))]

    num_workers = os.cpu_count() if not os.environ['DEBUG_MODE'] else 1
    with tqdm(total=len(chunks), desc="total: ", position=0) as pbar:
        with Pool(num_workers) as p:
            for result in p.imap_unordered(engine.process_batch, file_chunks):
                results.append(result)
    results = list(itertools.chain.from_iterable([r.items() for r in results]))
    results = [(r[0], r[1][0], r[1][1], r[1][2], r[1][3]) for r in results]
    with open("result_dump.json", "w") as f:
        json.dump(results, f)
    # with open("result_dump.json", "r") as f:
    #     results = json.load(f)

    plot_result_confusion_matrix(keywords, results)


def test_asr():
    data_folder = "/mnt/data/datasets/LibriSpeech/dev-clean"

    data_folder_kw = "/mnt/data/pytorch-kaldi/bench_data/speech_commands_v0.02"

    _, keywords = get_files_speech_commands(data_folder_kw, "validation_list.txt")

    files = get_files_librispeech(data_folder)
    files = files[:40]
    keywords = [kw.upper() for kw in keywords]
    # print(keywords)

    engine = KWSEngine(keywords, 0.0,
                       "/mnt/data/pytorch-kaldi/exp/libri_MLP_fbank_20190225_133944/checkpoints/checkpoint-epoch7.pth")

    results = engine.process_batch(files)
    plot_result_confusion_matrix(keywords, results)


if __name__ == '__main__':
    # test_kws()
    # test_asr()
    run_benchmark(os.path.abspath("bench_output"))
