import configparser
import os.path
import random
import subprocess
import json

import numpy as np
import torch
import torch.optim as optim
import matplotlib

from nets.TIMIT_LSTM import TIMIT_LSTM
from nets.losses.mtl_mono_cd_loss import MtlMonoCDLoss
from nets.metrics.metrics import LabCDAccuracy, LabMonoAccuracy

matplotlib.use('agg')
import matplotlib.pyplot as plt
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


def run_shell(cmd, logger):
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


def get_posterior_norm_data(config, logger):
    train_dataset_lab = config['datasets'][config['data_use']['train_with']]['lab']
    N_out_lab = {}

    for forward_out in config['test']:
        normalize_with_counts_from = config['test'][forward_out]['normalize_with_counts_from']
        if config['test'][forward_out]['normalize_posteriors']:
            # Try to automatically retrieve the config file
            assert "ali-to-pdf" in train_dataset_lab[normalize_with_counts_from]['lab_opts']
            folder_lab_count = train_dataset_lab[normalize_with_counts_from]['lab_folder']
            cmd = "hmm-info " + folder_lab_count + "/final.mdl | awk '/pdfs/{print $4}'"
            output = run_shell(cmd, logger)
            N_out = int(output.decode().rstrip())
            N_out_lab[normalize_with_counts_from] = N_out
            count_file_path = os.path.join(config['exp']['save_dir'], config['exp']['name'],
                                           'exp_files/forward_' + forward_out + '_' + \
                                           normalize_with_counts_from + '.count')
            cmd = "analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim=" + str(
                N_out) + " \"ark:ali-to-pdf " + folder_lab_count + "/final.mdl \\\"ark:gunzip -c " + folder_lab_count + "/ali.*.gz |\\\" ark:- |\" " + count_file_path
            run_shell(cmd, logger)
            config['test'][forward_out]['normalize_with_counts_from_file'] = count_file_path

    return config, N_out_lab

    # TODO check config


def model_init(arch_name, fea_index_length, lab_cd_num, use_cuda=False, multi_gpu=False):
    if arch_name == "TIMIT_LSTM":
        net = TIMIT_LSTM(inp_dim=fea_index_length, lab_cd_num=lab_cd_num)

    else:
        raise ValueError

    return net


def optimizer_init(config, trainable_params):
    if config['training']['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(trainable_params, **config['training']['optimizer']["args"])
    else:
        raise ValueError

    return optimizer


def loss_init(config):
    if config['arch']['loss']['name'] == 'mtl_mono_cd':
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    else:
        raise ValueError


def metrics_init(config):
    metrics = {}
    for metric in config['arch']['metrics']:
        if metric == 'acc_lab_cd':
            metrics[metric] = LabCDAccuracy()
        elif metric == 'acc_lab_mono':
            metrics[metric] = LabMonoAccuracy()
        else:
            raise ValueError("Can't find the metric {}".format(metric))
    return metrics


def lr_scheduler_init(config, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                mode='min',
                                                factor=config['training']['lr_scheduler']['arch_halving_factor'],
                                                patience=1,
                                                verbose=True,
                                                threshold=config['training']['lr_scheduler'][
                                                    'arch_improvement_threshold'],
                                                threshold_mode='rel')


def export_loss_acc_to_txt(out_folder, N_ep, val_lst):
    if not os.path.exists(out_folder + '/generated_outputs'):
        os.makedirs(out_folder + '/generated_outputs')

    nb_val = len(val_lst)
    res = open(out_folder + '/res.res', 'r').readlines()

    tr_loss = []
    tr_acc = []
    val_loss = np.ndarray((nb_val, N_ep))
    val_acc = np.ndarray((nb_val, N_ep))

    line_cpt = 0
    for i in range(N_ep):
        splitted = res[i].split(' ')

        # Getting uniq training loss and acc
        tr_loss.append(float(splitted[2].split('=')[1]))
        tr_acc.append(1 - float(splitted[3].split('=')[1]))

        # Getting multiple or uniq val loss and acc
        # +5 to avoird the 6 first columns of the res.res file
        for i in range(nb_val):
            val_loss[i][line_cpt] = float(splitted[(i * 3) + 5].split('=')[1])
            val_acc[i][line_cpt] = 1 - float(splitted[(i * 3) + 6].split('=')[1])

        line_cpt += 1

    # Saving to files
    np.savetxt(out_folder + '/generated_outputs/tr_loss.txt', np.asarray(tr_loss), '%0.3f', delimiter=',')
    np.savetxt(out_folder + '/generated_outputs/tr_acc.txt', np.asarray(tr_acc), '%0.3f', delimiter=',')

    for i in range(nb_val):
        np.savetxt(out_folder + '/generated_outputs/val_' + str(i) + '_loss.txt', val_loss[i], '%0.5f', delimiter=',')
        np.savetxt(out_folder + '/generated_outputs/val_' + str(i) + '_acc.txt', val_acc[i], '%0.5f', delimiter=',')


def create_curves(out_folder, N_ep, val_lst):
    print(' ')
    print('-----')
    print('Generating output files and plots ... ')
    export_loss_acc_to_txt(out_folder, N_ep, val_lst)

    if not os.path.exists(out_folder + '/generated_outputs'):
        raise RuntimeError('accOR: No results generated please call export_loss_err_to_txt() before')

    nb_epoch = len(open(out_folder + '/generated_outputs/tr_loss.txt', 'r').readlines())
    x = np.arange(nb_epoch)
    nb_val = len(val_lst)

    # Loading train Loss and acc
    tr_loss = np.loadtxt(out_folder + '/generated_outputs/tr_loss.txt')
    tr_acc = np.loadtxt(out_folder + '/generated_outputs/tr_acc.txt')

    # Loading val loss and acc
    val_loss = []
    val_acc = []
    for i in range(nb_val):
        val_loss.append(np.loadtxt(out_folder + '/generated_outputs/val_' + str(i) + '_loss.txt'))
        val_acc.append(np.loadtxt(out_folder + '/generated_outputs/val_' + str(i) + '_acc.txt'))

    #
    # LOSS PLOT
    #

    # Getting maximum values
    max_loss = np.amax(tr_loss)
    for i in range(nb_val):
        if np.amax(val_loss[i]) > max_loss:
            max_loss = np.amax(val_loss[i])

    # Plot train loss and acc
    plt.plot(x, tr_loss, label="train_loss")

    # Plot val loss and acc
    for i in range(nb_val):
        plt.plot(x, val_loss[i], label='val_' + str(i) + '_loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Evolution of the loss function')
    plt.axis([0, nb_epoch - 1, 0, max_loss + 1])
    plt.legend()
    plt.savefig(out_folder + '/generated_outputs/loss.png')

    # Clear plot
    plt.gcf().clear()

    #
    # ACC PLOT
    #

    # Plot train loss and acc
    plt.plot(x, tr_acc, label="train_acc")

    # Plot val loss and acc
    for i in range(nb_val):
        plt.plot(x, val_acc[i], label='val_' + str(i) + '_acc')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Evolution of the accuracy')
    plt.axis([0, nb_epoch - 1, 0, 1])
    plt.legend()
    plt.savefig(out_folder + '/generated_outputs/acc.png')

    print('OK')
