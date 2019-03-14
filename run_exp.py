import json
import os
import argparse
import datetime
import shutil

import torch

from base.utils import set_seed
from cfg.dataset_definition.get_dataset_definition import get_dataset_definition
from data.phoneme_dict import get_phoneme_dict
from nn_ import model_init, optimizer_init, lr_scheduler_init, metrics_init, loss_init
from nn_.registries.seq_len_scheduler_regestry import seq_len_scheduler_init
from trainer.trainer import Trainer
from utils.logger_config import logger
from utils.nvidia_smi import nvidia_smi_enabled
from utils.util import code_versioning
from utils.utils import check_environment, read_json


def check_config(config):
    # TODO impl schema or sth
    pass


def make_phn_dict(config, dataset_definition, label_name):
    if dataset_definition['data_info']['labels'][label_name]['lab_count'][0] < 1:
        # TODO counts return 0 indexed but phns are not so wtf
        dataset_definition['data_info']['labels'][label_name]['lab_count'] = \
            dataset_definition['data_info']['labels'][label_name]['lab_count'][1:]

    if len(config['dataset']['labels_use']) == 0:
        raise NotImplementedError("Multiple output labels not implemented for e2e/ctc")

    # e2e ctc
    phoneme_dict = get_phoneme_dict(config['dataset']['dataset_definition']['phn_mapping_file'],
                                    stress_marks=False, word_position_dependency=False)
    dataset_definition['data_info']['labels'][label_name]['num_lab'] = len(phoneme_dict.phoneme2reducedIdx)
    lab_count_new = [0] * len(phoneme_dict.phoneme2reducedIdx)
    for lab_idx, count in enumerate(dataset_definition['data_info']['labels'][label_name]['lab_count'], start=1):
        lab_count_new[phoneme_dict.idx2reducedIdx[lab_idx]] += count
    dataset_definition['data_info']['labels'][label_name]['lab_count'] = lab_count_new

    return phoneme_dict


def setup_run(config, optim_overwrite):
    set_seed(config['exp']['seed'])
    torch.backends.cudnn.deterministic = True  # Otherwise I got nans for the CTC gradient

    # TODO remove the data meta info part and move into kaldi folder e.g.
    dataset_definition = get_dataset_definition(config['dataset']['name'], config['dataset']['data_use']['train_with'])
    config['dataset']['dataset_definition'] = dataset_definition

    if 'lab_phn' in config['dataset']['labels_use']:
        phoneme_dict = make_phn_dict(config, dataset_definition, 'lab_phn')
    elif 'lab_phnframe' in config['dataset']['labels_use']:
        phoneme_dict = make_phn_dict(config, dataset_definition, 'lab_phnframe')
    else:
        # framewise
        phoneme_dict = get_phoneme_dict(config['dataset']['dataset_definition']['phn_mapping_file'],
                                        stress_marks=True, word_position_dependency=True)

    del config['dataset']['dataset_definition']['phn_mapping_file']
    config['dataset']['dataset_definition']['phoneme_dict'] = phoneme_dict

    model = model_init(config)

    optimizers, lr_schedulers = optimizer_init(config, model, optim_overwrite)

    seq_len_scheduler = seq_len_scheduler_init(config)

    logger.info("".join(["="] * 80))
    logger.info("Architecture:")
    logger.info(model)
    logger.info("".join(["="] * 80))
    metrics = metrics_init(config, model)

    loss = loss_init(config, model)

    return model, loss, metrics, optimizers, config, lr_schedulers, seq_len_scheduler


def main(config_path, load_path, restart, overfit_small_batch, warm_start, optim_overwrite):
    config = read_json(config_path)
    check_config(config)
    if optim_overwrite:
        optim_overwrite = read_json('cfg/optim_overwrite.json')

    if load_path is not None:
        raise NotImplementedError

    # if resume_path:
    # TODO
    #     resume_config = torch.load(folder_to_checkpoint(args.resume), map_location='cpu')['config']
    #     # also the results won't be the same give the different random seeds with different number of draws
    #     del config['exp']['name']
    #     recursive_update(resume_config, config)
    #
    #     print("".join(["="] * 80))
    #     print("Resume with these changes in the config:")
    #     print("".join(["-"] * 80))
    #     print(jsondiff.diff(config, resume_config, dump=True, dumper=jsondiff.JsonDumper(indent=1)))
    #     print("".join(["="] * 80))
    #
    #     config = resume_config
    #     # start_time = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    #     # config['exp']['name'] = config['exp']['name'] + "r-" + start_time
    # else:
    save_time = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    # config['exp']['name'] = config['exp']['name'] + start_time

    set_seed(config['exp']['seed'])

    config['exp']['save_dir'] = os.path.abspath(config['exp']['save_dir'])

    # Output folder creation
    out_folder = os.path.join(config['exp']['save_dir'], config['exp']['name'])
    if os.path.exists(out_folder):
        print(f"Experiement under {out_folder} exists, moving it copying it to backup")
        if os.path.exists(os.path.join(out_folder, "checkpoints")) \
                and len(os.listdir(os.path.join(out_folder, "checkpoints"))) > 0:
            shutil.copytree(out_folder,
                            os.path.join(config['exp']['save_dir'] + "_finished_runs_backup/",
                                         config['exp']['name'] + save_time))

        #     print(os.listdir(os.path.join(out_folder, "checkpoints")))
        #     resume_path = out_folder
        # else:
        if restart:
            shutil.rmtree(out_folder)
            os.makedirs(out_folder + '/exp_files')
    else:
        os.makedirs(out_folder + '/exp_files')

    logger.configure_logger(out_folder)

    check_environment()

    if nvidia_smi_enabled:  # TODO chage criteria or the whole thing
        git_commit = code_versioning()
        if 'versioning' not in config:
            config['versioning'] = {}
        config['versioning']['git_commit'] = git_commit

    logger.info("Experiment name : {}".format(out_folder))
    logger.info("tensorboard : tensorboard --logdir {}".format(os.path.abspath(out_folder)))

    model, loss, metrics, optimizers, config, lr_schedulers, seq_len_scheduler = setup_run(config, optim_overwrite)

    if warm_start is not None:
        load_warm_start_op = getattr(model, "load_warm_start", None)
        assert callable(load_warm_start_op)
        model.load_warm_start(warm_start)

    # TODO instead of resuming and making a new folder, make a backup and continue in the same folder
    trainer = Trainer(model, loss, metrics, optimizers, lr_schedulers, seq_len_scheduler,
                      load_path, config,
                      restart_optim=bool(optim_overwrite),
                      do_validation=True,
                      overfit_small_batch=overfit_small_batch)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--restart', action='store_true',
                        help='restart training from scratch')

    parser.add_argument('-l', '--load', default=None, type=str,
                        help='load specific checkpoint(default: None)')

    parser.add_argument('-w', '--warm_start', default=None, type=str,
                        help='path to checkpoint to load weights from for warm start (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-o', '--overfit', action='store_true',
                        help='overfit_small_batch / debug mode')

    parser.add_argument('--optim', action='store_true',
                        help='overwrite optim config and reinit optim')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.config, args.load, args.restart, args.overfit, args.warm_start, args.optim)
