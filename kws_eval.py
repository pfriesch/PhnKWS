import os
import random
from glob import glob

from kws_decoder.kws_engine import KWSEngine
import numpy as np
import matplotlib.pyplot as plt


def get_files(data_folder, file_list):
    with open(os.path.join(data_folder, file_list), "r")as f:
        file_list = f.readlines()

    keywords = set()
    files = []
    for file in file_list:
        file = file.strip()
        keywords.add(os.path.dirname(file))
        files.append(os.path.join(data_folder, file))

    return files, list(keywords)


def test():
    data_folder = "/mnt/data/libs/kaldi/egs/google_speech_commands/kws/data_kws/speech_commands_v0.02"

    files, keywords = get_files(data_folder, "validation_list.txt")
    random.shuffle(files)
    files = files[:200]
    keywords = [kw.upper() for kw in keywords]
    print(keywords)

    engine = KWSEngine(keywords, 0.0,
                       "/mnt/data/pytorch-kaldi/exp/libri_TDNN_fbank_20190206_175358/checkpoints/checkpoint-epoch6.pth")

    results = engine.process_batch(files)

    #### confusion_matrix
    confusion_matrix = np.zeros((len(keywords) + 1, len(keywords) + 1))
    keywords = ["<UNK>"] + keywords
    for sample_id, (transcript, lattice_confidence, lm_posterior, acoustic_posterior) in results.items():
        transcript = transcript[0]
        gt = sample_id.split("_", 1)[0].upper()
        if gt not in keywords:
            gt = "<UNK>"

        gt_index = keywords.index(gt)
        transcript_index = keywords.index(transcript)
        confusion_matrix[gt_index, transcript_index] += 1
        # print(transcript, gt)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.matshow(confusion_matrix)
    ax.set_xticklabels([''] + keywords)
    ax.set_yticklabels([''] + keywords)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("kw_resilt.png")
    plt.clf()
    # TODO plot the results against true/false results
    #### /confusion_matrix

    #### count
    count_gt = confusion_matrix.sum(axis=1)
    count_transcript = confusion_matrix.sum(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    b1 = ax.bar(np.arange(0, len(count_gt), dtype=float) - 0.25, count_gt, width=0.5, align='center')
    b2 = ax.bar(np.arange(0, len(count_transcript), dtype=float) + 0.25, count_transcript, width=0.5, align='center')
    ax.legend((b1, b2), ('count_gt', 'count_transcript'))
    plt.savefig("count_gt.png")
    plt.clf()

    #### count

    print(results)
    # plot_output_phonemes(model_logits)


if __name__ == '__main__':
    test()
