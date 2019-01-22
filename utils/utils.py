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


def phn_mapping(phone_path, no_triphone=True, no_spoken_noise=True, no_silence=True, no_eps=True, start_idx=1):
    phone_path = os.path.join(phone_path, "phones.txt")
    with open(phone_path, "r") as f:
        phones = f.readlines()

    def map_phone(phn, _id):
        if "#" in phn:
            phn_used = None
        elif no_silence and "SIL" in phn:
            phn_used = None
        elif no_spoken_noise and "SPN" in phn:
            phn_used = None
        elif no_eps and "<eps>" in phn:
            phn_used = None
        elif no_triphone:
            phn_used = phn.split("_")[0]
        else:
            phn_used = phn
        return phn, _id, phn_used

    def convert_phn(p):
        phn_str, phn_id = p.strip().split(" ")
        return map_phone(phn_str, int(phn_id))

    phn_all = [convert_phn(p) for p in phones]

    seen = set()
    seen_add = seen.add
    phn_all_ordered_set = [phn_used for phn, _id, phn_used in phn_all
                           if phn_used is not None and not (phn_used in seen or seen_add(phn_used))]

    phn_used_dict = {phn: _id for _id, phn in enumerate(phn_all_ordered_set, start=start_idx)}

    id_mapping = {id_true: phn_used_dict[phn_new] for phn_true, id_true, phn_new in phn_all if phn_new is not None}

    return {'all_phone_info': phn_all, 'used_dict': phn_used_dict,
            'id_mapping': id_mapping,
            'no_triphone': no_triphone,
            'no_spoken_noise': no_spoken_noise,
            'no_silence': no_silence,
            'no_eps': no_eps,
            'start_idx': start_idx}


def get_dataset_metadata(config):
    if 'test' in config:
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

    kw2phn_mapping = make_timit_kws_labels()

    config["kws_decoding"]["kw2phn_mapping"] = kw2phn_mapping

    _phn_mapping = {}
    label_dict = config['datasets'][config['data_use']['train_with']]['labels']
    for label_name in label_dict:
        _phn_mapping[label_name] = phn_mapping(label_dict[label_name]['label_folder'],
                                               no_triphone=True,
                                               no_spoken_noise=True,
                                               no_silence=True,
                                               no_eps=True)
    config['arch']['args']['phn_mapping'] = _phn_mapping
    if not config['arch']['framewise_labels']:
        # one for ctc blank symbol
        config['arch']['args']['lab_phn_num'] = len(_phn_mapping['lab_phn']['used_dict']) + 1

    return config


timit_kw2phn_mapping = \
    {'follow': "f aa1 l ow2",
     'eight': "ey1 t",
     'zero': None,
     'three': "th r iy1",
     'five': " f ay1 v",
     'right': "r ay1 t",
     'marvin': "m aa1 r v ix n",  # added by hand not in dict
     'no': "n ow1",
     'forward': "f ao1 r w axr d",
     'four': "f ao1 r",
     'learn': "l er1 n",
     'on': " ao1 n",
     'six': "s ih1 k s",
     'seven': "s eh1 v ax n",
     'happy': "hh ae1 p iy",
     'house': "hh aw1 s",
     'left': "l eh1 f t",
     'bird': "b er1 d",
     'up': "ah1 p",
     'one': "w ah1 n",
     'backward': "b ae1 k w er d z",
     'tree': "t r iy1",
     'stop': "s t aa1 p",
     'sheila': "sh iy1 l ax",
     'down': "d aw1 n",
     'yes': " y eh1 s",
     'off': "ao1 f",
     'wow': None,
     'bed': "b eh1 d",
     'cat': "k ae1 t",
     'visual': "v ih1 zh uw el",
     'two': "t uw1",
     'go': "g ow1",
     'dog': "d ao1 g",
     'nine': "n ay1 n"}

timit_phones = {
    # "<eps>": 0,
    "sil": 1,
    "aa": 2,
    "ae": 3,
    "ah": 4,
    "ao": 5,
    "aw": 6,
    "ax": 7,
    "ay": 8,
    "b": 9,
    "ch": 10,
    "cl": 11,
    "d": 12,
    "dh": 13,
    "dx": 14,
    "eh": 15,
    "el": 16,
    "en": 17,
    "epi": 18,
    "er": 19,
    "ey": 20,
    "f": 21,
    "g": 22,
    "hh": 23,
    "ih": 24,
    "ix": 25,
    "iy": 26,
    "jh": 27,
    "k": 28,
    "l": 29,
    "m": 30,
    "n": 31,
    "ng": 32,
    "ow": 33,
    "oy": 34,
    "p": 35,
    "r": 36,
    "s": 37,
    "sh": 38,
    "t": 39,
    "th": 40,
    "uh": 41,
    "uw": 42,
    "v": 43,
    "vcl": 44,
    "w": 45,
    "y": 46,
    "z": 47,
    "zh": 48,
    # "#0": 49,
    # "#1": 50,
}


def make_timit_kws_labels():
    res = {}

    def filter_phn(phn):
        phn = ''.join(i for i in phn if not i.isdigit())
        if phn == "axr":
            return "ax"
        else:
            return phn

    for text, phn in timit_kw2phn_mapping.items():
        if phn is not None:
            phn_no_num = [filter_phn(p) for p in phn.split(" ") if p is not '']
            phn_ids = [timit_phones[p] for p in phn_no_num]
            res[text] = {"phn_raw": phn, "phn_no_num": phn_no_num, "phn_ids": phn_ids}
        else:
            print("missing phones for {}, skipping...".format(text))
    return res
