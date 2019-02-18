import os.path
import random
import json
import subprocess

import numpy as np
import torch
import matplotlib

from data.data_util import load_counts
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
                         + f":{KALDI_ROOT}/src/kwsbin" \
                         + f":{os.environ['PATH']}"
    # TODO find better method

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

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash',
                         env=os.environ.copy())

    (output, err) = p.communicate()
    output = output.decode("utf-8")
    err = err.decode("utf-8")
    return_code = p.wait()
    if return_code > 0:
        logger.error(
            "Call:  \n{}\n{}\n{}\nReturn Code: {}\nstdout: {}\nstderr: {}"
                .format("".join(["-"] * 73), cmd, "".join(["-"] * 80), return_code, output, err))
        raise RuntimeError("Call: {} had nonzero return code: {}, stderr: {}".format(cmd, return_code, err))
    # logger.warn("ERROR: {}".format(err))

    logger.debug("OUTPUT: {}".format(output))
    return output

