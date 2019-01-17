import configparser
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


def which(program):
    """https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    """
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def check_environment():
    assert os.environ['KALDI_ROOT']

    PATH = os.environ['PATH']

    assert "tools/openfst" in PATH and "src/featbin" in PATH and "src/gmmbin" in PATH and "src/bin" in PATH and "src/nnetbin" in PATH

    assert isinstance(which("hmm-info"), str), which("hmm-info")
    # TODO test with which for other commands


def run_shell(cmd):
    logger.debug("RUN: {}".format(cmd))
    if cmd.split(" ")[0].endswith(".sh"):
        if not (os.path.isfile(cmd.split(" ")[0]) and os.access(cmd.split(" ")[0], os.X_OK)):
            logger.warn("{} does not exist or is not runnable!".format(cmd.split(" ")[0]))

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()
    return_code = p.wait()
    if return_code > 0:
        logger.error("Call: {} had nonzero return code: {}, stderr: {}".format(cmd, return_code, err))
        raise RuntimeError("Call: {} had nonzero return code: {}, stderr: {}".format(cmd, return_code, err))
    # logger.warn("ERROR: {}".format(err.decode("utf-8")))

    logger.debug("OUTPUT: {}".format(output.decode("utf-8")))
    return output


def compute_avg_performance(info_lst):
    losses = []
    errors = []
    times = []

    for tr_info_file in info_lst:
        config_res = configparser.ConfigParser()
        config_res.read(tr_info_file)
        losses.append(float(config_res['results']['loss']))
        errors.append(float(config_res['results']['err']))
        times.append(float(config_res['results']['elapsed_time']))

    loss = np.mean(losses)
    error = np.mean(errors)
    time = np.sum(times)

    return [loss, error, time]


def get_dataset_metadata(config):
    train_dataset_lab = config['datasets'][config['data_use']['train_with']]['labels']
    N_out_lab = {}

    for forward_out in config['test']:
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
            config['arch']['args']['lab_cd_num'] = N_out
