import logging
import os
import time
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from data.kaldi_dataset import KaldiDataset
from base.base_trainer import BaseTrainer
from data import kaldi_io
from data.data_util import load_counts
from data.kaldi_data_loader import KaldiDataLoader
from kaldi_decoding_scripts.decode_dnn import decode, best_wer
from utils.logger_config import logger


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizers, resume_path, config, do_validation,
                 lr_schedulers):
        super(Trainer, self).__init__(model, loss, metrics, optimizers, lr_schedulers, resume_path, config)
        self.config = config
        self.do_validation = do_validation
        self.log_step = int(np.sqrt(config['training']['batch_size_train']))

    def _eval_metrics(self, output, target):
        acc_metrics = {}
        for metric in self.metrics:
            acc_metrics[metric] = self.metrics[metric](output, target)
            self.tensorboard_logger.add_scalar(metric, acc_metrics[metric])
        return acc_metrics

    def _train_epoch(self, epoch, global_step):
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
        tr_data = self.config['data_use']['train_with']

        train_loss = 0
        train_metrics = {metric: 0 for metric in self.metrics}

        dataset = KaldiDataset(self.config['exp']['cache_data_root'],
                               tr_data,
                               self.config['datasets'][tr_data]['features'],
                               self.config['datasets'][tr_data]['labels'],
                               self.config['training']['max_seq_length_train'],
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['data_use']['phones'],
                               split_files_max_seq_len=self.max_seq_length_train_curr,
                               shuffle_frames=self.config['training']['shuffle_frames'])

        dataloader = KaldiDataLoader(dataset,
                                     self.config['training']['batch_size_train'],
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'])
        if self.max_seq_length_train_curr is not None:
            self.tensorboard_logger.add_scalar("max_seq_length_train_curr", self.max_seq_length_train_curr, epoch)

        accumulated_train_losses = {}
        accumulated_train_metrics = {}
        n_steps_chunk = 0
        last_train_logging = time.process_time()
        last_checkpoint = time.process_time()

        n_steps_this_epoch = 0
        # TODO chunked dataloader length
        with tqdm(disable=not logger.isEnabledFor(logging.INFO)) as pbar:
            pbar.set_description('T e:{} l: {} a: {}'.format(epoch, '-', '-'))
            for batch_idx, (_, inputs, targets) in enumerate(dataloader):
                global_step += 1
                n_steps_this_epoch += 1

                # TODO assert out.shape[1] >= lab_dnn.max() and lab_dnn.min() >= 0, \
                #     "lab_dnn max of {} is bigger than shape of output {} or min {} is smaller than 0" \
                #         .format(lab_dnn.max().cpu().numpy(), out.shape[1], lab_dnn.min().cpu().numpy())

                inputs = self.to_device(inputs)
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

                _train_metrics = self._eval_metrics(output, targets)
                for metric, metric_value in _train_metrics.items():
                    if metric not in accumulated_train_metrics:
                        accumulated_train_metrics[metric] = 0
                    accumulated_train_metrics[metric] += metric_value
                #
                pbar.set_description('T e:{} l: {:.4f}'.format(epoch,
                                                               loss["loss_final"].item()))
                pbar.update()

                # Log training every minute
                if (time.process_time() - last_train_logging) > 60:
                    self.tensorboard_logger.set_step(global_step, 'train')
                    for _loss, loss_value in accumulated_train_losses.items():
                        self.tensorboard_logger.add_scalar(_loss, loss_value / n_steps_chunk)
                    for metric, metric_value in accumulated_train_metrics.items():
                        self.tensorboard_logger.add_scalar(metric, metric_value / n_steps_chunk)

                    most_recent_inputs = inputs
                    for feat_name in most_recent_inputs:
                        if isinstance(most_recent_inputs[feat_name], dict) and 'sequence_lengths' in most_recent_inputs[
                            feat_name]:
                            total_padding = torch.sum(
                                (torch.ones_like(most_recent_inputs[feat_name]['sequence_lengths']) *
                                 most_recent_inputs[feat_name]['sequence_lengths'][0]) - most_recent_inputs[feat_name][
                                    'sequence_lengths'])
                            self.tensorboard_logger.add_scalar('total_padding_{}'.format(feat_name),
                                                               total_padding.item())

                    accumulated_train_losses = {}
                    accumulated_train_metrics = {}
                    n_steps_chunk = 0

                    if (time.process_time() - last_checkpoint) > self.config['exp']['checkpoint_interval_seconds']:
                        pass  # TODO add saving checkpoint

                #### /Logging ####

        self.tensorboard_logger.set_step(epoch, 'train')
        self.tensorboard_logger.add_scalar('train_loss_avg', train_loss / n_steps_this_epoch)
        for metric in train_metrics:
            self.tensorboard_logger.add_scalar(metric + "_avg", train_metrics[metric] / n_steps_this_epoch)

        log = {'train_loss_avg': train_loss / n_steps_this_epoch,
               'train_metrics_avg':
                   {metric: train_metrics[metric] / n_steps_this_epoch
                    for metric in train_metrics}}
        if self.do_validation:
            valid_log = self._valid_epoch(epoch, global_step=global_step)
            log.update(valid_log)

        return log, global_step

    def _valid_epoch(self, epoch, global_step):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        valid_data = self.config['data_use']['valid_with']

        valid_loss = 0
        valid_metrics = {metric: 0 for metric in self.metrics}

        dataset = KaldiDataset(self.config['exp']['cache_data_root'],
                               valid_data,
                               self.config['datasets'][valid_data]['features'],
                               self.config['datasets'][valid_data]['labels'],
                               self.config['training']['max_seq_length_valid'],
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['data_use']['phones'],
                               split_files_max_seq_len=self.max_seq_length_train_curr,
                               shuffle_frames=self.config['training']['shuffle_frames'])

        dataloader = KaldiDataLoader(dataset,
                                     self.config['training']['batch_size_valid'],
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'])

        n_steps_this_epoch = 0
        with tqdm(disable=not logger.isEnabledFor(logging.INFO)) as pbar:
            pbar.set_description('V e:{} l: {} '.format(epoch, '-'))
            for batch_idx, (_, inputs, targets) in enumerate(dataloader):
                n_steps_this_epoch += 1

                inputs = self.to_device(inputs)
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

    def _eval_epoch(self, epoch, global_step):

        if 'test' in self.config:
            result_decode = self._eval_epoch_kaldi_decode(epoch, global_step)
        else:
            result_decode = self._eval_epoch_ctc_decode(epoch, global_step)
        result_kws = self._eval_kws(epoch, global_step)
        return {"result_decode": result_decode, "result_kws": result_kws}

    def _eval_epoch_ctc_decode(self, epoch, global_step):
        self.model.eval()
        batch_size = 1
        max_seq_length = -1

        test_data = self.config['data_use']['test_with']
        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        dataset = KaldiDataset(self.config['exp']['cache_data_root'],
                               test_data,
                               self.config['datasets'][test_data]['features'],
                               self.config['datasets'][test_data]['labels'],
                               max_seq_length,
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['data_use']['phones'],
                               split_files_max_seq_len=self.max_seq_length_train_curr,
                               shuffle_frames=self.config['training']['shuffle_frames'])

        dataloader = KaldiDataLoader(dataset,
                                     batch_size,
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'])

        test_metrics = {metric: 0 for metric in self.metrics}

        with tqdm(total=len(dataloader), disable=not logger.isEnabledFor(logging.INFO)) as pbar:
            pbar.set_description('E e:{}    '.format(epoch))
            for batch_idx, (_, inputs, targets) in tqdm(enumerate(dataloader)):
                inputs = self.to_device(inputs)
                targets = self.to_device(targets)

                output = self.model(inputs)

                #### Logging ####
                _eval_metrics = self._eval_metrics(output, targets)
                test_metrics = {metric: test_metrics[metric] + metric_value for
                                metric, metric_value
                                in _eval_metrics.items()}
                pbar.set_description(
                    'E e:{} a: {:.4f} '.format(epoch, test_metrics[self.config['arch']['metrics'][0]]))
                pbar.update()
                #### /Logging ####

        logger.critical("Done decoding... TODO implement with lm decoding")

        self.tensorboard_logger.set_step(global_step, 'test')
        for metric in test_metrics:
            self.tensorboard_logger.add_scalar(metric, test_metrics[metric] / len(dataloader))

        return {'test_metrics': {metric: test_metrics[metric] / len(dataloader) for metric in
                                 test_metrics}}

    def _eval_epoch_kaldi_decode(self, epoch, global_step):
        self.model.eval()
        batch_size = 1
        max_seq_length = -1

        test_data = self.config['data_use']['test_with']
        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        test_metrics = {metric: 0 for metric in self.metrics}

        dataset = KaldiDataset(self.config['exp']['cache_data_root'],
                               test_data,
                               self.config['datasets'][test_data]['features'],
                               self.config['datasets'][test_data]['labels'],
                               max_seq_length,
                               self.model.context_left,
                               self.model.context_right,
                               normalize_features=True,
                               phoneme_dict=self.config['data_use']['phones'],
                               split_files_max_seq_len=self.max_seq_length_train_curr,
                               shuffle_frames=self.config['training']['shuffle_frames'])

        dataloader = KaldiDataLoader(dataset,
                                     batch_size,
                                     self.config["exp"]["n_gpu"] > 0,
                                     self.config['exp']['num_workers'])

        with KaldiOutputWriter(out_folder, test_data, epoch, self.config) as writer:

            with tqdm(total=len(dataloader), disable=not logger.isEnabledFor(logging.INFO)) as pbar:
                pbar.set_description('E e:{}    '.format(epoch))
                for batch_idx, (sample_names, inputs, targets) in tqdm(enumerate(dataloader)):
                    inputs = self.to_device(inputs)
                    targets = self.to_device(targets)

                    output = self.model(inputs)

                    warned_label = False
                    for output_label in output:
                        if output_label in self.config['test'].keys():
                            # squeeze that batch
                            output[output_label] = output[output_label].squeeze(1)
                            # remove blank/padding 0th dim
                            if self.config["arch"]["framewise_labels"] == "shuffled_frames":
                                out_save = output[output_label].data.cpu().numpy()
                            else:
                                out_save = output[output_label][:, :-1].data.cpu().numpy()

                            if len(out_save.shape) == 3 and out_save.shape[0] == 1:
                                out_save = out_save.squeeze(0)

                            if output_label in self.config['test'] and \
                                    self.config['test'][output_label]['normalize_posteriors']:
                                # read the config file
                                counts = load_counts(
                                    self.config['test'][output_label]['normalize_with_counts_from_file'])
                                out_save = out_save - np.log(counts / np.sum(counts))

                                # save the output
                                # data_name = file ids
                                # out save shape <class 'tuple'>: (124, 1944)
                                # post_file dict out_dnn2: buffered wirter
                            assert len(out_save.shape) == 2
                            assert len(sample_names) == 1
                            writer.write_mat(output_label, out_save.squeeze(), sample_names[0])

                            #### Logging ####
                            _test_metrics = self._eval_metrics(output, targets)
                            test_metrics = {metric: test_metrics[metric] + metric_value for
                                            metric, metric_value
                                            in _test_metrics.items()}

                            pbar.set_description('E e:{}           '.format(epoch))
                            pbar.update()
                            #### /Logging ####
                        else:
                            if not warned_label:
                                logger.debug("Skipping saving forward for decoding for key {}".format(output_label))
                                warned_label = True

            self.tensorboard_logger.set_step(global_step, 'eval')
            for metric, metric_value in test_metrics.items():
                self.tensorboard_logger.add_scalar(metric, test_metrics[metric] / len(dataloader))

        test_metrics = {metric: test_metrics[metric] / len(dataloader)
                        for metric in test_metrics}

        decoding_results = []
        #### DECODING ####
        for out_lab in self.config['test']:

            # forward_data_lst = self.config['data_use']['test_with'] #TODO multiple forward sets
            forward_data_lst = [self.config['data_use']['test_with']]
            # forward_dec_outs = self.config['test'][out_lab]['require_decoding']

            for data in forward_data_lst:
                logger.debug('Decoding {} output {}'.format(data, out_lab))

                lab_field = self.config['datasets'][data]['labels']['lab_cd']

                out_folder = os.path.abspath(out_folder)
                out_dec_folder = '{}/decode_{}_{}'.format(out_folder, data, out_lab)

                files_dec_list = glob('{}/exp_files/forward_{}_ep*_{}_to_decode.ark'.format(out_folder, data, out_lab))

                decode(**self.config['decoding'],
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
        else:
            return data.to(self.device)


class KaldiOutputWriter:

    def __init__(self, out_folder, test_data, epoch, config):
        super().__init__()
        self.out_folder = out_folder
        self.test_data = test_data
        self.epoch = epoch
        self.config = config

    def __enter__(self):
        base_file_name = '{}/exp_files/forward_{}_ep{:03d}'.format(self.out_folder, self.test_data, self.epoch)
        self.post_file = {}
        for out_name in self.config['test'].keys():
            if self.config['test'][out_name]['require_decoding']:
                out_file = '{}_{}_to_decode.ark'.format(base_file_name, out_name)
            else:
                out_file = '{}_{}.ark'.format(base_file_name, out_name)

            self.post_file[out_name] = kaldi_io.open_or_fd(out_file, 'wb')
        return self

    def __exit__(self, *args):
        for out_name in self.config['test'].keys():
            self.post_file[out_name].close()

    def write_mat(self, out_name, out_save, sample_name):
        kaldi_io.write_mat(self.post_file[out_name], out_save, sample_name)
