import logging
import time
import torch

from tqdm import tqdm

from base.base_trainer import to_device, detach, eval_metrics
from base.utils import save_checkpoint
from data.dataset_regestry import get_dataset
from data.kaldi_data_loader import KaldiDataLoader
from utils.logger_config import logger


def train_epoch(epoch, global_step, model, loss_fun, metrics, config, max_label_length, device, tensorboard_logger,
                seq_len_scheduler, overfit_small_batch, starting_dataset_sampler_state, optimizers, lr_schedulers,
                do_validation, _valid_epoch, checkpoint_dir):
    """
    Training logic for an epoch

    :param epoch: Current training epoch.
    :return: A log that contains all information you want to save.

    Note:
        If you have additional information to record, for example:
            > additional_log = {"x": x, "y": y}
        merge it with log before return. i.e.
            > log = {**log, **additional_log}
            > return log

        The metrics in log must have the key 'metrics'.
    """
    model.train()
    tensorboard_logger.set_step(global_step, 'train')
    tr_data = config['dataset']['data_use']['train_with']
    _all_feats = config['dataset']['dataset_definition']['datasets'][tr_data]['features']
    _all_labs = config['dataset']['dataset_definition']['datasets'][tr_data]['labels']

    dataset = get_dataset(config['training']['dataset_type'],
                          config['exp']['data_cache_root'],
                          f"{tr_data}_{config['exp']['name']}",
                          {feat: _all_feats[feat] for feat in config['dataset']['features_use']},
                          {lab: _all_labs[lab] for lab in config['dataset']['labels_use']},
                          config['training']['batching']['max_seq_length_train'],
                          model.context_left,
                          model.context_right,
                          normalize_features=True,
                          phoneme_dict=config['dataset']['dataset_definition']['phoneme_dict'],
                          max_seq_len=seq_len_scheduler.get_seq_len(epoch),
                          max_label_length=max_label_length,
                          overfit_small_batch=overfit_small_batch)

    dataloader = KaldiDataLoader(dataset,
                                 config['training']['batching']['batch_size_train'],
                                 config["exp"]["n_gpu"] > 0,
                                 batch_ordering=model.batch_ordering,
                                 shuffle=True)

    if starting_dataset_sampler_state is not None:
        dataloader.sampler.load_state_dict(starting_dataset_sampler_state)
        starting_dataset_sampler_state = None

    assert len(dataset) >= config['training']['batching']['batch_size_train'], \
        f"Length of train dataset {len(dataset)} too small " \
        + f"for batch_size of {config['training']['batching']['batch_size_train']}"

    total_train_loss = 0
    total_train_metrics = {metric: 0 for metric in metrics}

    accumulated_train_losses = {}
    accumulated_train_metrics = {metric: 0 for metric in metrics}
    n_steps_chunk = 0
    last_train_logging = time.time()
    last_checkpoint = time.time()

    n_steps_this_epoch = 0

    with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader), position=0) as pbar:
        pbar.set_description('T e:{} l: {} a: {}'.format(epoch, '-', '-'))
        pbar.update(dataloader.start())
        # TODO remove for epoch after 0

        for batch_idx, (_, inputs, targets) in enumerate(dataloader):
            global_step += 1
            n_steps_this_epoch += 1

            # TODO assert out.shape[1] >= lab_dnn.max() and lab_dnn.min() >= 0, \
            #     "lab_dnn max of {} is bigger than shape of output {} or min {} is smaller than 0" \
            #         .format(lab_dnn.max().cpu().numpy(), out.shape[1], lab_dnn.min().cpu().numpy())

            inputs = to_device(device, inputs)
            if "lab_phn" not in targets:
                targets = to_device(device, targets)

            for opti in optimizers.values():
                opti.zero_grad()

            with torch.autograd.detect_anomaly():
                # TODO check if there is a perfomance penalty
                output = model(inputs)
                loss = loss_fun(output, targets)
                loss["loss_final"].backward()

            if config['training']['clip_grad_norm'] > 0:
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                torch.nn.utils.clip_grad_norm_(trainable_params, config['training']['clip_grad_norm'])
            for opti in optimizers.values():
                opti.step()

            # detach so metrics etc. don't accumulate gradients
            inputs = detach(inputs)
            targets = detach(targets)
            loss = detach(loss)

            #### Logging ####
            n_steps_chunk += 1
            for _loss, loss_value in loss.items():
                if _loss not in accumulated_train_losses:
                    accumulated_train_losses[_loss] = 0
                accumulated_train_losses[_loss] += loss_value
            total_train_loss += loss["loss_final"]

            if config['exp']['compute_train_metrics']:
                """
                If the metric computation is fast like with plain accuracy on a discrete output, it is better to 
                perform it in a batched fashion on the GPU. 
                The alternative would be to copy the result (blocking) to the CPU and then compute 
                the metrics asynchronously (not batched).
                On the otherhand, if the metrics computation is not implemented on GPU or does not benefit from
                batching that much, it is preferred to copy the result (blocking) to the CPU and then compute 
                the metrics asynchronously (not batched).
                | main thread |  metrics thread |
                =================================
                     |        '
                forward pass  '
                     |        '        
                     +---> output -> comput metric -----+
                     |        '                         |
                forward pass  '                         +-> accumulate metrics
                     |        '                         |
                     +---> output -> comput metric -----+
                     |        '
                forward pass  '
                     |        '

                """

                _train_metrics = eval_metrics((output, targets), metrics)
                for metric, metric_value in _train_metrics.items():
                    accumulated_train_metrics[metric] += metric_value
                    total_train_metrics[metric] += metric_value

            pbar.set_description('T e:{} l: {:.4f}'.format(epoch,
                                                           loss["loss_final"].item()))
            pbar.update()

            # Log training every 30s and smoothe since its the average
            if (time.time() - last_train_logging) > 30:
                # TODO add flag for delayed logging
                last_train_logging = time.time()
                tensorboard_logger.set_step(global_step, 'train')
                for _loss, loss_value in accumulated_train_losses.items():
                    tensorboard_logger.add_scalar(_loss, loss_value / n_steps_chunk)

                if config['exp']['compute_train_metrics']:
                    for metric, metric_value in accumulated_train_metrics.items():
                        tensorboard_logger.add_scalar(metric, metric_value / n_steps_chunk)

                # most_recent_inputs = inputs
                # for feat_name in most_recent_inputs:
                #     if isinstance(most_recent_inputs[feat_name], dict) \
                #             and 'sequence_lengths' in most_recent_inputs[feat_name]:
                #         total_padding = torch.sum(
                #             (torch.ones_like(most_recent_inputs[feat_name]['sequence_lengths'])
                #              * most_recent_inputs[feat_name]['sequence_lengths'][0])
                #             - most_recent_inputs[feat_name]['sequence_lengths'])
                #         tensorboard_logger.add_scalar('total_padding_{}'.format(feat_name),
                #                                            total_padding.item())

                accumulated_train_losses = {}
                if config['exp']['compute_train_metrics']:
                    accumulated_train_metrics = {metric: 0 for metric in metrics}
                n_steps_chunk = 0

                if (time.time() - last_checkpoint) > config['exp']['checkpoint_interval_minutes'] * 60:
                    save_checkpoint(epoch, global_step, model, optimizers, lr_schedulers,
                                    seq_len_scheduler, config, checkpoint_dir,
                                    dataset_sampler_state=dataloader.sampler.state_dict())

                    last_checkpoint = time.time()

                #### /Logging ####

            del inputs
            del targets

    if n_steps_this_epoch > 0:
        tensorboard_logger.set_step(epoch, 'train')
        tensorboard_logger.add_scalar('train_loss_avg', total_train_loss / n_steps_this_epoch)
        if config['exp']['compute_train_metrics']:
            for metric in total_train_metrics:
                tensorboard_logger.add_scalar(metric + "_avg",
                                              total_train_metrics[metric] / n_steps_this_epoch)

        # TODO add this flag to vlaid since ctcdecode is fucking slow or do it async
        if config['exp']['compute_train_metrics']:
            log = {'train_loss_avg': total_train_loss / n_steps_this_epoch,
                   'train_metrics_avg':
                       {metric: total_train_metrics[metric] / n_steps_this_epoch
                        for metric in total_train_metrics}}
        else:
            log = {'train_loss_avg': total_train_loss / n_steps_this_epoch}
        if do_validation and (not overfit_small_batch or epoch == 1):
            valid_log = _valid_epoch(epoch)
            log.update(valid_log)
        else:
            log.update({'valid_loss': -1,
                        'valid_metrics': {}})
    else:
        raise RuntimeError("Training epoch hat 0 batches.")

    return log, starting_dataset_sampler_state, global_step
