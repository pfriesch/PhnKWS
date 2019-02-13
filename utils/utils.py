import os.path
import random
import json
import subprocess

import numpy as np
import torch
import matplotlib

from utils.logger_config import logger

matplotlib.use('agg')
from jsmin import jsmin


def set_seed(seed):
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # TODO add saving and loading of random state for reproducable research


def config2dict(config):
    return {s: dict(config.items(s)) for s in config.sections()}


def read_json(path):
    if not (os.path.exists(path)):
        raise ValueError('ERROR: The json file {} does not exist!\n'.format(path))
    else:
        with open(path, "r") as js_file:
            _dict = json.loads(jsmin(js_file.read()))
    return _dict


def write_json(_dict, path, overwrite=False):
    assert isinstance(_dict, dict)
    if not overwrite:
        assert not os.path.exists(path), "path {} exits and overwrite is false".format(path)
    with open(path, "w") as js_file:
        json.dump(_dict, js_file, indent=1)


def check_environment():
    assert os.environ['KALDI_ROOT']
    KALDI_ROOT = os.environ['KALDI_ROOT']
    PATH = os.environ['PATH']
    # TODO find better method
    run_shell(
        f"export PATH={KALDI_ROOT}/src/bin/:{KALDI_ROOT}/tools/openfst/bin/:{KALDI_ROOT}/tools/irstlm/bin/:{KALDI_ROOT}/src/fstbin/:{KALDI_ROOT}/src/gmmbin/:{KALDI_ROOT}/src/featbin/:{KALDI_ROOT}/src/lm/:{KALDI_ROOT}/src/lmbin/:{KALDI_ROOT}/src/sgmmbin/:{KALDI_ROOT}/src/sgmm2bin/:{KALDI_ROOT}/src/fgmmbin/:{KALDI_ROOT}/src/latbin/:{KALDI_ROOT}/src/nnetbin/:{KALDI_ROOT}/src/nnet2bin/:{KALDI_ROOT}/src/kwsbin:{PATH}")

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


def run_shell(cmd, pipefail=True):
    """


    :param cmd:
    :param pipefail:    From bash man: If pipefail is enabled, the pipeline's return status is
     the value of the last (rightmost) command to exit with a non-zero status, or zero if all
     commands exit successfully.
    :return:
    """
    logger.debug("RUN: {}".format(cmd))
    if cmd.split(" ")[0].endswith(".sh"):
        if not (os.path.isfile(cmd.split(" ")[0]) and os.access(cmd.split(" ")[0], os.X_OK)):
            logger.warn("{} does not exist or is not runnable!".format(cmd.split(" ")[0]))

    if pipefail:
        cmd = 'set -o pipefail; ' + cmd

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

    (output, err) = p.communicate()
    return_code = p.wait()
    if return_code > 0:
        logger.error(
            "Call: {} had nonzero return code: {}, stdout: {} stderr: {}".format(cmd, return_code, output, err))
        raise RuntimeError("Call: {} had nonzero return code: {}, stderr: {}".format(cmd, return_code, err))
    # logger.warn("ERROR: {}".format(err.decode("utf-8")))

    logger.debug("OUTPUT: {}".format(output.decode("utf-8")))
    return output


def get_dataset_metadata(config):
    if 'test' in config:
        train_dataset_lab = config['datasets'][config['data_use']['train_with']]['labels']
        N_out_lab = {}
        for forward_out in config['test']:
            if 'normalize_with_counts_from' in config['test'][forward_out]:
                normalize_with_counts_from = config['test'][forward_out]['normalize_with_counts_from']
                assert 'label_opts' in train_dataset_lab[normalize_with_counts_from]
                if config['test'][forward_out]['normalize_posteriors']:
                    # Try to automatically retrieve the config file
                    assert "ali-to-pdf" in train_dataset_lab[normalize_with_counts_from]['label_opts']
                    folder_lab_count = train_dataset_lab[normalize_with_counts_from]['label_folder']
                    cmd = "hmm-info " + folder_lab_count + "/final.mdl | awk '/pdfs/{print $4}'"
                    output = run_shell(cmd)
                    N_out = int(output.decode().rstrip())
                    N_out_lab[normalize_with_counts_from] = N_out
                    count_file_path = os.path.join(config['exp']['save_dir'], config['exp']['name'],
                                                   'exp_files/forward_' + forward_out + '_' + \
                                                   normalize_with_counts_from + '.count')
                    cmd = "analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim=" + str(
                        N_out) + " \"ark:ali-to-pdf " + folder_lab_count + "/final.mdl \\\"ark:gunzip -c " + folder_lab_count + "/ali.*.gz |\\\" ark:- |\" " + count_file_path
                    run_shell(cmd)
                    config['test'][forward_out]['normalize_with_counts_from_file'] = count_file_path
                    # kaldi labels are indexed at 1
                    # we use 0 for padding or blank label
                    config['arch']['args']['lab_cd_num'] = N_out

                    # if not config["arch"]["framewise_labels"]:
                    config['arch']['args']['lab_cd_num'] += 1
            else:
                logger.debug("Skipping getting normalize_with_counts for {}".format(forward_out))

    if "lab_mono_num" in config['arch']['args']:
        config['arch']['args']['lab_mono_num'] += 1

    return config
