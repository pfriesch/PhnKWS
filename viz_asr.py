import json
import logging
import os
import argparse
import datetime
import shutil
import numpy as np
import torch

import ctcdecode
from tqdm import tqdm

from base.base_trainer import to_device, detach_cpu, eval_metrics
from base.utils import set_seed
from cfg.dataset_definition.get_dataset_definition import get_dataset_definition
from data.dataset_regestry import get_dataset
from data.kaldi_data_loader import KaldiDataLoader
from data.phoneme_dict import get_phoneme_dict
from nn_ import model_init, optimizer_init, lr_scheduler_init, metrics_init, loss_init
from nn_.registries.seq_len_scheduler_regestry import seq_len_scheduler_init
from trainer.trainer import Trainer
from utils.logger_config import logger
from utils.nvidia_smi import nvidia_smi_enabled
from utils.util import code_versioning
from utils.utils import check_environment, read_json, sample_id_to_transcript, plot_alignment_spectrogram_ctc




def valid_epoch_sync_metrics(epoch, model, loss_fun, metrics, config, max_label_length, device, tensorboard_logger):
    model.eval()

    valid_loss = 0
    accumulated_valid_metrics = {metric: 0 for metric in metrics}

    valid_data = config['dataset']['data_use']['valid_with']
    _all_feats = config['dataset']['dataset_definition']['datasets'][valid_data]['features']
    _all_labs = config['dataset']['dataset_definition']['datasets'][valid_data]['labels']
    dataset = get_dataset(config['training']['dataset_type'],
                          config['exp']['data_cache_root'],
                          f"{valid_data}_{config['exp']['name']}",
                          {feat: _all_feats[feat] for feat in config['dataset']['features_use']},
                          {lab: _all_labs[lab] for lab in config['dataset']['labels_use']},
                          config['training']['batching']['max_seq_length_valid'],
                          model.context_left,
                          model.context_right,
                          normalize_features=True,
                          phoneme_dict=config['dataset']['dataset_definition']['phoneme_dict'],
                          max_seq_len=config['training']['batching']['max_seq_length_valid'],
                          max_label_length=max_label_length)

    if config['training']['batching']['batch_size_valid'] != 1:
        logger.warn("setting valid batch size to 1 to avoid padding zeros")
    dataloader = KaldiDataLoader(dataset,
                                 config['training']['batching']['batch_size_valid'],
                                 config["exp"]["n_gpu"] > 0,
                                 batch_ordering=model.batch_ordering)

    assert len(dataset) >= config['training']['batching']['batch_size_valid'], \
        f"Length of valid dataset {len(dataset)} too small " \
        + f"for batch_size of {config['training']['batching']['batch_size_valid']}"

    n_steps_this_epoch = 0
    with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader)) as pbar:
        pbar.set_description('V e:{} l: {} '.format(epoch, '-'))
        for batch_idx, (sample_name, inputs, targets) in enumerate(dataloader):
            n_steps_this_epoch += 1

            inputs = to_device(device, inputs)
            if "lab_phn" not in targets:
                targets = to_device(device, targets)

            output = model(inputs)
            loss = loss_fun(output, targets)

            output = detach_cpu(output)
            targets = detach_cpu(targets)
            loss = detach_cpu(loss)

            #### Logging ####
            valid_loss += loss["loss_final"].item()
            _valid_metrics = eval_metrics((output, targets), metrics)
            for metric, metric_value in _valid_metrics.items():
                accumulated_valid_metrics[metric] += metric_value

            pbar.set_description('V e:{} l: {:.4f} '.format(epoch, loss["loss_final"].item()))
            pbar.update()

            do_plotting = True
            if n_steps_this_epoch == 60 or n_steps_this_epoch == 1 and do_plotting:
                # raise NotImplementedError("TODO: add plots to tensorboard")
                output = output['out_phn']
                inputs = inputs["fbank"].numpy()
                _phoneme_dict = dataset.state.phoneme_dict
                vocabulary_size = len(dataset.state.phoneme_dict.reducedIdx2phoneme) + 1
                vocabulary = [chr(c) for c in list(range(65, 65 + 58)) + list(range(65 + 58 + 69, 65 + 58 + 69 + 500))][
                             :vocabulary_size]
                decoder = ctcdecode.CTCBeamDecoder(vocabulary, log_probs_input=True, beam_width=1)

                decoder_logits = output.permute(0, 2, 1)
                # We expect batch x seq x label_size
                beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(decoder_logits)

                _targets = []
                curr_l = 0
                for l in targets['target_sequence_lengths']:
                    _targets.append(targets['lab_phn'][curr_l:curr_l + l])
                    curr_l += l
                for i in range(len(inputs)):
                    _beam_result = beam_result[i, 0, :out_seq_len[i, 0]]
                    # logger.debug(sample_name)
                    result_decoded = [_phoneme_dict.reducedIdx2phoneme[l.item() - 1] for l in _beam_result]
                    result_decoded = " ".join(result_decoded)
                    logger.debug("RES: " + result_decoded)
                    # plot_phns = True
                    # if plot_phns:
                    label_decoded = " ".join(
                        [_phoneme_dict.reducedIdx2phoneme[l.item() - 1] for l in _targets[i]])
                    logger.debug("LAB: " + label_decoded)
                    text = sample_id_to_transcript(sample_name[i], "/mnt/data/datasets/LibriSpeech/dev-clean")
                    logger.debug("TXT: " + text)

                    # if plot_phns:
                    plot_alignment_spectrogram_ctc(sample_name[i], inputs[i],
                                                   (np.exp(output.numpy()[i]).T / np.exp(output.numpy()[i]).sum(
                                                       axis=1)).T,
                                                   _phoneme_dict, label_decoded, text,
                                                   result_decoded=result_decoded)
                    # else:
                    #     plot_alignment_spectrogram(sample_name, inputs["fbank"][i],
                    #                                (np.exp(output).T / np.exp(output).sum(axis=1)).T,
                    #                                _phoneme_dict, result_decoded=result_decoded)

            #### /Logging ####
    for metric, metric_value in accumulated_valid_metrics.items():
        accumulated_valid_metrics[metric] += metric_value

    tensorboard_logger.set_step(epoch, 'valid')
    tensorboard_logger.add_scalar('valid_loss', valid_loss / n_steps_this_epoch)
    for metric in accumulated_valid_metrics:
        tensorboard_logger.add_scalar(metric, accumulated_valid_metrics[metric] / n_steps_this_epoch)

    return {'valid_loss': valid_loss / n_steps_this_epoch,
            'valid_metrics': {metric: accumulated_valid_metrics[metric] / n_steps_this_epoch for metric in
                              accumulated_valid_metrics}}


def check_config(config):
    # TODO impl schema or sth
    pass


# def make_phn_dict(config, dataset_definition, label_name):
#     if dataset_definition['data_info']['labels'][label_name]['lab_count'][0] < 1:
#         # TODO counts return 0 indexed but phns are not so wtf
#         dataset_definition['data_info']['labels'][label_name]['lab_count'] = \
#             dataset_definition['data_info']['labels'][label_name]['lab_count'][1:]
#
#     if len(config['dataset']['labels_use']) == 0:
#         raise NotImplementedError("Multiple output labels not implemented for e2e/ctc")
#
#     # e2e ctc
#     phoneme_dict = get_phoneme_dict(config['dataset']['dataset_definition']['phn_mapping_file'],
#                                     stress_marks=False, word_position_dependency=False)
#     dataset_definition['data_info']['labels'][label_name]['num_lab'] = len(phoneme_dict.phoneme2reducedIdx)
#     lab_count_new = [0] * len(phoneme_dict.phoneme2reducedIdx)
#     for lab_idx, count in enumerate(dataset_definition['data_info']['labels'][label_name]['lab_count'], start=1):
#         lab_count_new[phoneme_dict.idx2reducedIdx[lab_idx]] += count
#     dataset_definition['data_info']['labels'][label_name]['lab_count'] = lab_count_new
#
#     return phoneme_dict


def setup_run(config, optim_overwrite):
    set_seed(config['exp']['seed'])
    torch.backends.cudnn.deterministic = True  # Otherwise I got nans for the CTC gradient

    # TODO remove the data meta info part and move into kaldi folder e.g.
    # dataset_definition = get_dataset_definition(config['dataset']['name'], config['dataset']['data_use']['train_with'])
    # config['dataset']['dataset_definition'] = dataset_definition
    #
    # if 'lab_phn' in config['dataset']['labels_use']:
    #     phoneme_dict = make_phn_dict(config, dataset_definition, 'lab_phn')
    # elif 'lab_phnframe' in config['dataset']['labels_use']:
    #     phoneme_dict = make_phn_dict(config, dataset_definition, 'lab_phnframe')
    # else:
    #     # framewise
    #     phoneme_dict = get_phoneme_dict(config['dataset']['dataset_definition']['phn_mapping_file'],
    #                                     stress_marks=True, word_position_dependency=True)
    #
    # del config['dataset']['dataset_definition']['phn_mapping_file']
    # config['dataset']['dataset_definition']['phoneme_dict'] = phoneme_dict

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


def main(warm_start):
    config = torch.load(warm_start, map_location='cpu')['config']
    config['exp']['name'] = config['exp']['name'] + "_viz_eval"

    # config = read_json(config_path)
    # check_config(config)
    # if optim_overwrite:
    #     optim_overwrite = read_json('cfg/optim_overwrite.json')

    # if load_path is not None:
    #     raise NotImplementedError

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
    out_folder = os.path.join(config['exp']['save_dir'], config['exp']['name'] + "_eval")
    if os.path.exists(out_folder):
        pass
        # print(f"Experiement under {out_folder} exists, moving it copying it to backup")
        # if os.path.exists(os.path.join(out_folder, "checkpoints")) \
        #         and len(os.listdir(os.path.join(out_folder, "checkpoints"))) > 0:
        #     shutil.copytree(out_folder,
        #                     os.path.join(config['exp']['save_dir'] + "_finished_runs_backup/",
        #                                  config['exp']['name'] + save_time))

        #     print(os.listdir(os.path.join(out_folder, "checkpoints")))
        #     resume_path = out_folder
        # else:
        # if restart:
        #     shutil.rmtree(out_folder)
        #     os.makedirs(out_folder + '/exp_files')
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

    model, loss, metrics, optimizers, config, lr_schedulers, seq_len_scheduler = setup_run(config, False)

    if warm_start is not None:
        load_warm_start_op = getattr(model, "load_warm_start", None)
        assert callable(load_warm_start_op)
        model.load_warm_start(warm_start)

    # TODO instead of resuming and making a new folder, make a backup and continue in the same folder
    trainer = Trainer(model, loss, metrics, optimizers, lr_schedulers, seq_len_scheduler,
                      None, config,
                      restart_optim=False,
                      do_validation=True,
                      overfit_small_batch=False)
    log = trainer._valid_epoch(-1)
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', default=None, type=str,
    #                     help='config file path (default: None)')
    parser.add_argument('-w', '--warm_start', type=str,
                        help='path to checkpoint to load weights from for warm start (default: None)')
    args = parser.parse_args()

    # if args.device:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.warm_start)
