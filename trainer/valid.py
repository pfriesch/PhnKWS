import logging
import os
from multiprocessing import Manager
from multiprocessing.pool import Pool

from tqdm import tqdm

from base.base_trainer import metrics_accumulator, to_device, detach_cpu, eval_metrics
from data.dataset_regestry import get_dataset
from data.kaldi_data_loader import KaldiDataLoader
from utils.logger_config import logger


def valid_epoch_async_metrics(epoch, model, loss_fun, metrics, config, max_label_length, device, tensorboard_logger):
    """
    Validate after training an epoch
    :return: A log that contains information about validation
    Note:
        The validation metrics in log must have the key 'val_metrics'.
    """
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

    dataloader = KaldiDataLoader(dataset,
                                 config['training']['batching']['batch_size_valid'],
                                 config["exp"]["n_gpu"] > 0,
                                 batch_ordering=model.batch_ordering)

    assert len(dataset) >= config['training']['batching']['batch_size_valid'], \
        f"Length of valid dataset {len(dataset)} too small " \
        + f"for batch_size of {config['training']['batching']['batch_size_valid']}"

    n_steps_this_epoch = 0

    with Pool(os.cpu_count()) as pool:
        multip_process = Manager()
        metrics_q = multip_process.Queue(maxsize=os.cpu_count())
        # accumulated_valid_metrics_future_list = pool.apply_async(metrics_accumulator, (metrics_q, metrics))
        accumulated_valid_metrics_future_list = [pool.apply_async(metrics_accumulator, (metrics_q, metrics))
                                                 for _ in range(os.cpu_count())]
        with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader)) as pbar:
            pbar.set_description('V e:{} l: {} '.format(epoch, '-'))
            for batch_idx, (_, inputs, targets) in enumerate(dataloader):
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
                metrics_q.put((output, targets))
                # _valid_metrics = eval_metrics((output, targets), metrics)
                # for metric, metric_value in _valid_metrics.items():
                #     accumulated_valid_metrics[metric] += metric_value

                pbar.set_description('V e:{} l: {:.4f} '.format(epoch, loss["loss_final"].item()))
                pbar.update()
                #### /Logging ####
        for _accumulated_valid_metrics in accumulated_valid_metrics_future_list:
            metrics_q.put(None)
        for _accumulated_valid_metrics in accumulated_valid_metrics_future_list:
            _accumulated_valid_metrics = _accumulated_valid_metrics.get()
            for metric, metric_value in _accumulated_valid_metrics.items():
                accumulated_valid_metrics[metric] += metric_value

    tensorboard_logger.set_step(epoch, 'valid')
    tensorboard_logger.add_scalar('valid_loss', valid_loss / n_steps_this_epoch)
    for metric in accumulated_valid_metrics:
        tensorboard_logger.add_scalar(metric, accumulated_valid_metrics[metric] / n_steps_this_epoch)

    return {'valid_loss': valid_loss / n_steps_this_epoch,
            'valid_metrics': {metric: accumulated_valid_metrics[metric] / n_steps_this_epoch for metric in
                              accumulated_valid_metrics}}


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
        for batch_idx, (_, inputs, targets) in enumerate(dataloader):
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
