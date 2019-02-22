import logging
import os
import time
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

# from data.kaldi_dataset_prefetch_gpu import KaldiDataset
from data.kaldi_dataset import KaldiDataset
from base.base_trainer import BaseTrainer
from data import kaldi_io
from data.kaldi_data_loader import KaldiDataLoader
from kaldi_decoding_scripts.decode_dnn import decode, best_wer
from nn_.losses.ctc_phn import CTCPhnLoss
from utils.logger_config import logger


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizers, lr_schedulers,
                 resume_path, config,
                 do_validation, overfit_small_batch):
        super(Trainer, self).__init__(model, loss, metrics, optimizers, lr_schedulers,
                                      resume_path, config)
        self.config = config
        self.do_validation = do_validation
        self.log_step = int(np.sqrt(config['training']['batch_size_train']))
        self.overfit_small_batch = overfit_small_batch
        # Necessary for cudnn ctc function
        self.max_label_length = 256 if isinstance(self.loss, CTCPhnLoss) else None

    def _eval_metrics(self, output, target):
        acc_metrics = {}
        for metric in self.metrics:
            acc_metrics[metric] = self.metrics[metric](output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
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
        self.model.train()
        self.tensorboard_logger.set_step(self.global_step, 'train')
        tr_data = self.config['dataset']['data_use']['train_with']
        _all_feats = self.config['dataset']['dataset_definition']['datasets'][tr_data]['features']
        _all_labs = self.config['dataset']['dataset_definition']['datasets'][tr_data]['labels']

        dataset = KaldiDataset(self.config['exp']['data_cache_root'], tr_data,
                               {feat: _all_feats[feat] for feat in self.config['dataset']['features_use']},
                               {lab: _all_labs[lab] for lab in self.config['dataset']['labels_use']},
                               self.device,
                               self.config['training']['max_seq_length_train'],
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['dataset']['dataset_definition']['phn_mapping_file'],
                               max_seq_len=self.max_seq_length_train_curr,
                               max_label_length=self.max_label_length,
                               shuffle_frames=self.config['training']['shuffle_frames'],
                               overfit_small_batch=self.overfit_small_batch)

        dataloader = KaldiDataLoader(dataset,
                                     self.config['training']['batch_size_train'],
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'],
                                     shuffle=True)

        assert len(dataset) >= self.config['training']['batch_size_train'], \
            f"Length of train dataset {len(dataset)} too small " \
            + f"for batch_size of {self.config['training']['batch_size_train']}"

        total_train_loss = 0
        total_train_metrics = {metric: 0 for metric in self.metrics}

        accumulated_train_losses = {}
        accumulated_train_metrics = {}
        n_steps_chunk = 0
        last_train_logging = time.time()
        last_checkpoint = time.time()

        n_steps_this_epoch = 0
        # TODO chunked dataloader length
        with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader)) as pbar:
            pbar.set_description('T e:{} l: {} a: {}'.format(epoch, '-', '-'))
            for batch_idx, (_, inputs, targets) in enumerate(dataloader):
                self.global_step += 1
                n_steps_this_epoch += 1

                # TODO assert out.shape[1] >= lab_dnn.max() and lab_dnn.min() >= 0, \
                #     "lab_dnn max of {} is bigger than shape of output {} or min {} is smaller than 0" \
                #         .format(lab_dnn.max().cpu().numpy(), out.shape[1], lab_dnn.min().cpu().numpy())

                inputs = self.to_device(inputs)
                if "lab_phn" not in targets:
                    targets = self.to_device(targets)

                for opti in self.optimizers.values():
                    opti.zero_grad()

                with torch.autograd.detect_anomaly():
                    output = self.model(inputs)
                    loss = self.loss(output, targets)
                    loss["loss_final"].backward()

                if self.config['training']['clip_grad_norm'] > 0:
                    trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config['training']['clip_grad_norm'])
                for opti in self.optimizers.values():
                    opti.step()

                #### Logging ####
                n_steps_chunk += 1
                for _loss, loss_value in loss.items():
                    if _loss not in accumulated_train_losses:
                        accumulated_train_losses[_loss] = 0
                    accumulated_train_losses[_loss] += loss_value
                total_train_loss += loss["loss_final"]

                if self.config['exp']['compute_train_metrics']:
                    _train_metrics = self._eval_metrics(output, targets)
                    for metric, metric_value in _train_metrics.items():
                        if metric not in accumulated_train_metrics:
                            accumulated_train_metrics[metric] = 0
                        accumulated_train_metrics[metric] += metric_value
                        total_train_metrics[metric] += metric_value
                #
                pbar.set_description('T e:{} l: {:.4f}'.format(epoch,
                                                               loss["loss_final"].item()))
                pbar.update()

                # Log training every 30s and smoothe since its the average
                if (time.time() - last_train_logging) > 30:
                    last_train_logging = time.time()
                    self.tensorboard_logger.set_step(self.global_step, 'train')
                    for _loss, loss_value in accumulated_train_losses.items():
                        self.tensorboard_logger.add_scalar(_loss, loss_value / n_steps_chunk)
                    if self.config['exp']['compute_train_metrics']:
                        for metric, metric_value in accumulated_train_metrics.items():
                            self.tensorboard_logger.add_scalar(metric, metric_value / n_steps_chunk)

                    most_recent_inputs = inputs
                    for feat_name in most_recent_inputs:
                        if isinstance(most_recent_inputs[feat_name], dict) \
                                and 'sequence_lengths' in most_recent_inputs[feat_name]:
                            total_padding = torch.sum(
                                (torch.ones_like(most_recent_inputs[feat_name]['sequence_lengths'])
                                 * most_recent_inputs[feat_name]['sequence_lengths'][0])
                                - most_recent_inputs[feat_name]['sequence_lengths'])
                            self.tensorboard_logger.add_scalar('total_padding_{}'.format(feat_name),
                                                               total_padding.item())

                    accumulated_train_losses = {}
                    if self.config['exp']['compute_train_metrics']:
                        accumulated_train_metrics = {}
                    n_steps_chunk = 0

                    if (time.time() - last_checkpoint) > self.config['exp']['checkpoint_interval_seconds']:
                        logger.warn("checkpoint_interval_seconds not implemented")
                        # TODO add saving checkpoint

                #### /Logging ####

        if n_steps_this_epoch > 0:
            self.tensorboard_logger.set_step(epoch, 'train')
            self.tensorboard_logger.add_scalar('train_loss_avg', total_train_loss / n_steps_this_epoch)
            if self.config['exp']['compute_train_metrics']:
                for metric in total_train_metrics:
                    self.tensorboard_logger.add_scalar(metric + "_avg",
                                                       total_train_metrics[metric] / n_steps_this_epoch)

            if self.config['exp']['compute_train_metrics']:
                log = {'train_loss_avg': total_train_loss / n_steps_this_epoch,
                       'train_metrics_avg':
                           {metric: total_train_metrics[metric] / n_steps_this_epoch
                            for metric in total_train_metrics}}
            else:
                log = {'train_loss_avg': total_train_loss / n_steps_this_epoch}
            if self.do_validation:
                valid_log = self._valid_epoch(epoch)
                log.update(valid_log)
        else:
            raise RuntimeError("Training epoch hat 0 batches.")

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        valid_loss = 0
        valid_metrics = {metric: 0 for metric in self.metrics}

        valid_data = self.config['dataset']['data_use']['valid_with']
        _all_feats = self.config['dataset']['dataset_definition']['datasets'][valid_data]['features']
        _all_labs = self.config['dataset']['dataset_definition']['datasets'][valid_data]['labels']
        dataset = KaldiDataset(self.config['exp']['data_cache_root'],
                               valid_data,
                               {feat: _all_feats[feat] for feat in self.config['dataset']['features_use']},
                               {lab: _all_labs[lab] for lab in self.config['dataset']['labels_use']},
                               self.device,
                               self.config['training']['max_seq_length_valid'],
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['dataset']['dataset_definition']['phn_mapping_file'],
                               max_seq_len=self.config['training']['max_seq_length_valid'],
                               max_label_length=self.max_label_length,
                               shuffle_frames=self.config['training']['shuffle_frames'])

        dataloader = KaldiDataLoader(dataset,
                                     self.config['training']['batch_size_valid'],
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'])

        assert len(dataset) >= self.config['training']['batch_size_valid'], \
            f"Length of valid dataset {len(dataset)} too small " \
            + f"for batch_size of {self.config['training']['batch_size_valid']}"

        n_steps_this_epoch = 0
        with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader)) as pbar:
            pbar.set_description('V e:{} l: {} '.format(epoch, '-'))
            for batch_idx, (_, inputs, targets) in enumerate(dataloader):
                n_steps_this_epoch += 1

                inputs = self.to_device(inputs)
                if "lab_phn" not in targets:
                    targets = self.to_device(targets)

                output = self.model(inputs)
                loss = self.loss(output, targets)

                #### Logging ####
                valid_loss += loss["loss_final"].item()
                _eval_metrics = self._eval_metrics(output, targets)
                valid_metrics = {metric: valid_metrics[metric] + metric_value for
                                 metric, metric_value
                                 in _eval_metrics.items()}
                pbar.set_description('V e:{} l: {:.4f} '.format(epoch, loss["loss_final"].item()))
                pbar.update()
                #### /Logging ####

        self.tensorboard_logger.set_step(epoch, 'valid')
        self.tensorboard_logger.add_scalar('valid_loss', valid_loss / n_steps_this_epoch)
        for metric in valid_metrics:
            self.tensorboard_logger.add_scalar(metric, valid_metrics[metric] / n_steps_this_epoch)

        return {'valid_loss': valid_loss / n_steps_this_epoch,
                'valid_metrics': {metric: valid_metrics[metric] / n_steps_this_epoch for metric in
                                  valid_metrics}}

    def _eval_epoch(self, epoch):
        self.model.eval()
        batch_size = 1
        max_seq_length = -1

        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        test_metrics = {metric: 0 for metric in self.metrics}

        test_data = self.config['dataset']['data_use']['test_with']
        _all_feats = self.config['dataset']['dataset_definition']['datasets'][test_data]['features']
        _all_labs = self.config['dataset']['dataset_definition']['datasets'][test_data]['labels']
        dataset = KaldiDataset(self.config['exp']['data_cache_root'],
                               test_data,
                               {feat: _all_feats[feat] for feat in self.config['dataset']['features_use']},
                               {lab: _all_labs[lab] for lab in self.config['dataset']['labels_use']},
                               self.device,
                               max_seq_length,
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['dataset']['dataset_definition']['phn_mapping_file'],
                               max_seq_len=max_seq_length,
                               max_label_length=self.max_label_length,
                               shuffle_frames=self.config['training']['shuffle_frames'])

        dataloader = KaldiDataLoader(dataset,
                                     batch_size,
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'])

        assert len(dataset) >= batch_size, \
            f"Length of valid dataset {len(dataset)} too small " \
            + f"for batch_size of {batch_size}"

        n_steps_this_epoch = 0
        warned_size = False
        with KaldiOutputWriter(out_folder, test_data, self.model.out_names, epoch, self.config) as writer:
            with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader)) as pbar:
                pbar.set_description('E e:{}    '.format(epoch))
                for batch_idx, (sample_names, inputs, targets) in enumerate(dataloader):
                    n_steps_this_epoch += 1

                    inputs = self.to_device(inputs)
                    if "lab_phn" not in targets:
                        targets = self.to_device(targets)

                    output = self.model(inputs)

                    #### Logging ####
                    _eval_metrics = self._eval_metrics(output, targets)
                    test_metrics = {metric: test_metrics[metric] + metric_value for
                                    metric, metric_value
                                    in _eval_metrics.items()}

                    pbar.set_description('E e:{}           '.format(epoch))
                    pbar.update()
                    #### /Logging ####

                    warned_label = False
                    for output_label in output:
                        if output_label in self.model.out_names:
                            # squeeze that batch
                            output[output_label] = output[output_label].squeeze(1)
                            # remove blank/padding 0th dim
                            if self.config["arch"]["framewise_labels"] == "shuffled_frames":
                                out_save = output[output_label].data.cpu().numpy()
                            else:
                                raise NotImplementedError("TODO make sure the right dimension is taken")
                                out_save = output[output_label][:, :-1].data.cpu().numpy()

                            if len(out_save.shape) == 3 and out_save.shape[0] == 1:
                                out_save = out_save.squeeze(0)

                            if self.config['dataset']['dataset_definition']['decoding']['normalize_posteriors']:
                                # read the config file
                                counts = self.config['dataset']['dataset_definition'] \
                                    ['data_info']['labels']['lab_phn']['lab_count']
                                if out_save.shape[-1] == len(counts) - 1:
                                    if not warned_size:
                                        logger.info(
                                            f"Counts length is {len(counts)} but output has size {out_save.shape[-1]}."
                                            + f" Assuming that counts is 1 indexed")
                                        warned_size = True
                                    counts = counts[1:]
                                # Normalize by output count
                                if ctc:
                                    blank_scale = 1.0
                                    # TODO try different blank_scales 4.0 5.0 6.0 7.0
                                    counts[0] /= blank_scale
                                    # for i in range(1, 8):
                                    #     counts[i] /= noise_scale #TODO try noise_scale for SIL SPN etc I guess

                                prior = np.log(counts / np.sum(counts))

                                out_save = out_save - np.log(prior)

                            assert len(out_save.shape) == 2
                            assert len(sample_names) == 1
                            writer.write_mat(output_label, out_save.squeeze(), sample_names[0])


                        else:
                            if not warned_label:
                                logger.debug("Skipping saving forward for decoding for key {}".format(output_label))
                                warned_label = True

            self.tensorboard_logger.set_step(self.global_step, 'eval')
            for metric, metric_value in test_metrics.items():
                self.tensorboard_logger.add_scalar(metric, test_metrics[metric] / len(dataloader))

        test_metrics = {metric: test_metrics[metric] / len(dataloader)
                        for metric in test_metrics}

        decoding_results = []
        #### DECODING ####
        for out_lab in self.model.out_names:

            # forward_data_lst = self.config['data_use']['test_with'] #TODO multiple forward sets
            forward_data_lst = [self.config['dataset']['data_use']['test_with']]
            # forward_dec_outs = self.config['test'][out_lab]['require_decoding']

            for data in forward_data_lst:
                logger.debug('Decoding {} output {}'.format(data, out_lab))

                if out_lab == 'out_cd':
                    _label = 'lab_cd'
                elif out_lab == 'out_phn':
                    _label = 'lab_phn'
                else:
                    raise NotImplementedError(out_lab)

                lab_field = self.config['dataset']['dataset_definition']['datasets'][data]['labels'][_label]

                out_folder = os.path.abspath(out_folder)
                out_dec_folder = '{}/decode_{}_{}'.format(out_folder, data, out_lab)

                files_dec_list = glob('{}/exp_files/forward_{}_ep*_{}_to_decode.ark'.format(out_folder, data, out_lab))

                decode(**self.config['dataset']['dataset_definition']['decoding'],
                       alidir=os.path.abspath(lab_field['label_folder']),
                       data=os.path.abspath(lab_field['lab_data_folder']),
                       graphdir=os.path.abspath(lab_field['lab_graph']),
                       out_folder=out_dec_folder,
                       featstrings=files_dec_list)

                decoding_results = best_wer(out_dec_folder, self.config['decoding']['scoring_type'])
                logger.info(decoding_results)

                self.tensorboard_logger.add_text("WER results", str(decoding_results))

            # TODO plotting curves

        return {'test_metrics': test_metrics, "decoding_results": decoding_results}

    def to_device(self, data):
        if isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [e.to(self.device) for e in data]
        else:
            return data.to(self.device)


class KaldiOutputWriter:

    def __init__(self, out_folder, data_name, output_names, epoch, config):
        super().__init__()
        self.out_folder = out_folder
        self.data_name = data_name
        self.epoch = epoch
        self.config = config
        self.output_names = output_names

    def __enter__(self):
        base_file_name = '{}/exp_files/logits_{}_ep{:03d}'.format(self.out_folder, self.data_name, self.epoch)
        self.post_file = {}
        for out_name in self.output_names:
            out_file = '{}_{}.ark'.format(base_file_name, out_name)
            self.post_file[out_name] = kaldi_io.open_or_fd(out_file, 'wb')
        return self

    def __exit__(self, *args):
        for out_name in self.output_names:
            self.post_file[out_name].close()

    def write_mat(self, out_name, out_save, sample_name):
        kaldi_io.write_mat(self.post_file[out_name], out_save, sample_name)
