import os.path
import json
import subprocess
import logging

import numpy as np
import matplotlib

from utils.logger_config import logger

matplotlib.use('agg')
import matplotlib.pyplot as plt
from jsmin import jsmin
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def config2dict(config):
    return {s: dict(config.items(s)) for s in config.sections()}


def read_json(path):
    if not (os.path.exists(path)):
        raise ValueError(f'ERROR: The json file {path} does not exist!\n')
    else:
        with open(path, "r") as js_file:
            _dict = json.loads(jsmin(js_file.read()))
    return _dict


def write_json(_dict, path, overwrite=False):
    assert isinstance(_dict, dict)
    if not overwrite:
        assert not os.path.exists(path), f"path {path} exits and overwrite is false"
    with open(path, "w") as js_file:
        json.dump(_dict, js_file, indent=1)


def check_environment():
    assert os.environ['KALDI_ROOT']
    assert os.path.isdir(os.environ['KALDI_ROOT'])
    KALDI_ROOT = os.environ['KALDI_ROOT']
    os.environ['PATH'] = f"{KALDI_ROOT}/src/bin/" \
                         + f":{KALDI_ROOT}/tools/openfst/bin/" \
                         + f":{KALDI_ROOT}/tools/irstlm/bin/" \
                         + f":{KALDI_ROOT}/src/fstbin/" \
                         + f":{KALDI_ROOT}/src/gmmbin/" \
                         + f":{KALDI_ROOT}/src/featbin/" \
                         + f":{KALDI_ROOT}/src/lm/" \
                         + f":{KALDI_ROOT}/src/lmbin/" \
                         + f":{KALDI_ROOT}/src/sgmmbin/" \
                         + f":{KALDI_ROOT}/src/sgmm2bin/" \
                         + f":{KALDI_ROOT}/src/fgmmbin/" \
                         + f":{KALDI_ROOT}/src/latbin/" \
                         + f":{KALDI_ROOT}/src/nnetbin/" \
                         + f":{KALDI_ROOT}/src/nnet2bin/" \
                         + f":{KALDI_ROOT}/src/kwsbin/" \
                         + f":{os.environ['PATH']}"
    #### KALDI ####

    run_shell("which hmm-info")
    run_shell("which lattice-align-phones")
    run_shell("which lattice-to-ctm-conf")

    # Build FST
    run_shell("which fstcompile")
    run_shell("which fstaddselfloops")
    run_shell("which fstarcsort")

    run_shell("which fsttablecompose")
    run_shell("which fstdeterminizestar")
    run_shell("which fstminimizeencoded")

    run_shell("which arpa2fst")

    # Extract Feats
    run_shell("which compute-fbank-feats")
    run_shell("which copy-feats")

    # Decoding
    run_shell("which lattice-best-path")
    run_shell("which lattice-add-penalty")
    run_shell("which lattice-scale")

    #### /KALDI ####


def run_shell_info(cmd, stdin=None, pipefail=True):
    return run_shell(cmd, stdin, pipefail, cmd_logging_level=logging.INFO)


def run_shell(cmd, stdin=None, pipefail=True, cmd_logging_level=logging.DEBUG):
    """
    :param cmd:
    :param stdin:
    :param pipefail:    From bash man: If pipefail is enabled, the pipeline's return status is
     the value of the last (rightmost) command to exit with a non-zero status, or zero if all
     commands exit successfully.
    :return:
    """
    assert stdin is None or isinstance(stdin, bytes), f"Expected bytes as input for stdin, got {type(stdin)}"

    logger.log(cmd_logging_level, f"RUN: {cmd}")
    if cmd.split(" ")[0].endswith(".sh"):
        if not (os.path.isfile(cmd.split(" ")[0]) and os.access(cmd.split(" ")[0], os.X_OK)):
            logger.warn(f"{cmd.split(' ')[0]} does not exist or is not runnable!")

    if pipefail:
        cmd = 'set -o pipefail; ' + cmd

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash',
                         env=os.environ.copy())

    (output, err) = p.communicate(stdin)
    output = output.decode("utf-8")
    err = err.decode("utf-8")
    return_code = p.wait()
    if return_code > 0:
        logger.error(
            "Call:  \n{}\n{}\n{}\nReturn Code: {}\nstdout: {}\nstderr: {}"
                .format("".join(["-"] * 73), cmd, "".join(["-"] * 80), return_code, output, err))
        raise RuntimeError(f"Call: {cmd} had nonzero return code: {return_code}, stderr: {err}")
    # logger.warn("ERROR: {}".format(err))

    logger.log(cmd_logging_level, f"OUTPUT: {output}")
    return output


def plot_result_confusion_matrix(keywords, results):
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

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.matshow(confusion_matrix)
    # tickmar_font_dict = {'fontsize': 8}
    # ax.set_xticklabels([''] + keywords, fontdict=tickmar_font_dict)
    # ax.set_yticklabels([''] + keywords, fontdict=tickmar_font_dict)

    tick_marks = np.arange(len(keywords))
    plt.xticks(tick_marks, keywords, rotation=90, fontsize=8)
    assert keywords[0] == "<UNK>"
    plt.yticks(tick_marks, [" "] + keywords[1:], fontsize=8)

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

    # print(json.dumps(results, indent=1))
    # plot_output_phonemes(model_logits)


def feat_without_context(input_feat):
    _input_feat = input_feat.squeeze(1)
    out_feat = np.zeros((_input_feat.shape[0] + _input_feat.shape[2], _input_feat.shape[1]))
    for i in range(out_feat.shape[0]):
        if i >= _input_feat.shape[0]:
            out_feat[i] = _input_feat[_input_feat.shape[0] - 1, :, i - _input_feat.shape[0]]
        else:
            out_feat[i] = _input_feat[i, :, 0]

    return out_feat


def plot_alignment_spectrogram(sample_name, input_feat, output, phn_dict, _labels=None, result_decoded=None):
    min_height = 0.10
    top_phns = [x[0] for x in list(sorted(enumerate(output.max(axis=0)), key=lambda x: x[1], reverse=True))
                if output[:, x[0]].max() > min_height]

    if _labels is not None:
        _labels = _labels['lab_mono'][sample_name]
        _labels = [phn_dict.idx2phoneme[l] for l in _labels]
        prev_phn = None
        _l_out = []
        _l_out_i = []

        for _i, l in enumerate(_labels):
            if prev_phn is None:
                prev_phn = l
                # _l_out.append("")
            else:
                if prev_phn == l:
                    pass
                # _l_out.append("")
                else:
                    _l_out.append(prev_phn)
                    _l_out_i.append(_i)
                    prev_phn = l

    if 0 in top_phns:
        top_phns.remove(0)  # TODO removed blank maybe add later

    # phn_dict = {k + 1: v for k, v in phn_dict.items()}
    # phn_dict[0] = "<blk>"
    # assert len(phn_dict) == output.shape[1]

    height = 500

    fig = plt.figure()
    ax = fig.subplots()
    in_feat = feat_without_context(input_feat)
    ax.imshow(in_feat.T, origin='lower',
              # extent=[-(in_feat.shape[0] - output.shape[0] + 1) // 2, in_feat.shape[0], 0, 100],
              extent=[-(in_feat.shape[0] - output.shape[0]), in_feat.shape[0], 0, height],
              alpha=0.5)
    for i in top_phns:
        # ax.plot(output[:, i] * height, linewidth=0.5)
        if i != 0:
            # x = (output[:, i] * height).argmax()
            # y = (output[:, i] * height)[x]

            peaks, _ = find_peaks(output[:, i] * height, height=min_height * height, distance=10)
            # plt.plot(peaks, (output[:, i] * height)[peaks], "x", markersize=1)

            for peak in peaks:
                plt.axvline(x=peak, ymax=(output[:, i] * height)[peak] / height, linewidth=0.5, color='r',
                            linestyle='-')
                ax.annotate(phn_dict.reducedIdx2phoneme[i - 1], xy=(peak, (output[:, i] * height)[peak]), fontsize=4)
    # ax.
    if _labels is not None:
        ax.set_xticklabels(_l_out, rotation='vertical')
        ax.set_xticks(_l_out_i)
    # ax.legend()
    # ax.xaxis.set_major_locator(ticker.FixedLocator(_l_out_i))
    # ax.xaxis.set_(ticker.FixedLocator(_l_out_i))
    plt.tick_params(labelsize=4)
    ax.set_aspect(aspect=0.2)
    if result_decoded is None:
        ax.set_title(result_decoded)
    fig.savefig(f"output_{sample_name}.png")
    fig.savefig(f"output_{sample_name}.pdf")
    fig.clf()
    # plt.close(fig)


def get_open_fds():
    '''
    return the number of open file descriptors for current process

    .. warning: will only work on UNIX-like os-es.
    '''

    pid = os.getpid()
    procs = run_shell(f"lsof -w -Ff -p {pid}", cmd_logging_level=-1)

    def function(s):
        return s and s[0] == 'f' and s[1:].isdigit()

    nprocs = len(
        list(filter(
            function,
            procs.split('\n')))
    )
    return nprocs
