import configparser
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from data import kaldi_io
from data.data_util import load_counts
from data.kaldi_data_loader import KaldiDataLoader
from data.kaldi_dataset import KaldiDataset
from utils.logger_config import logger
from utils.utils import run_shell


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizers, resume_path, config, do_validation,
                 lr_schedulers, debug=False, local=False):
        super(Trainer, self).__init__(model, loss, metrics, optimizers, lr_schedulers, resume_path, config)
        self.config = config
        self.do_validation = do_validation
        self.log_step = int(np.sqrt(config['training']['batch_size_train']))
        self.debug = debug
        self.local = local

    def _eval_metrics(self, output, target):
        acc_metrics = {}
        for metric in self.metrics:
            acc_metrics[metric] = self.metrics[metric](output, target)
            self.tensorboard_logger.add_scalar(metric, acc_metrics[metric])
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
        self.tensorboard_logger.set_step((epoch - 1), 'train')
        tr_data = self.config['data_use']['train_with']

        train_loss = 0
        train_metrics = {metric: 0 for metric in self.metrics}

        dataset = KaldiDataset(self.config['datasets'][tr_data]['features'], self.config['datasets'][tr_data]['labels'],
                               self.model.context_left, self.model.context_right,
                               self.max_seq_length_train_curr,
                               self.tensorboard_logger, self.debug, self.local)
        data_loader = KaldiDataLoader(dataset,
                                      self.config['training']['batch_size_train'],
                                      use_gpu=self.config["exp"]["n_gpu"] > 0,
                                      prefetch_to_gpu=self.config['exp']['prefetch_to_gpu'],
                                      device=self.device,
                                      num_workers=0,
                                      sort_by_feat=self.config['training']['sort_by_feat'])

        with tqdm(total=len(data_loader), disable=not logger.isEnabledFor(logging.INFO)) as pbar:
            pbar.set_description('T e:{} l: {} a: {}'.format(epoch, '-', '-'))
            for batch_idx, (sample_names, inputs, targets) in enumerate(data_loader):

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}

                for opti in self.optimizers.values():
                    opti.zero_grad()
                output = self.model(inputs)
                loss = self.loss(output, targets)
                loss["loss_final"].backward()

                # TODO assert out.shape[1] >= lab_dnn.max() and lab_dnn.min() >= 0, \
                #     "lab_dnn max of {} is bigger than shape of output {} or min {} is smaller than 0" \
                #         .format(lab_dnn.max().cpu().numpy(), out.shape[1], lab_dnn.min().cpu().numpy())

                if self.config['training']['clip_grad_norm'] > 0:
                    trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config['training']['clip_grad_norm'])
                for opti in self.optimizers.values():
                    opti.step()

                #### Logging ####
                self.tensorboard_logger.set_step((epoch - 1) * len(data_loader) + batch_idx, 'train')
                for _loss, loss_value in loss.items():
                    self.tensorboard_logger.add_scalar(_loss, loss_value.item())
                train_loss += loss["loss_final"].item()
                _train_metrics = self._eval_metrics(output, targets)
                train_metrics = {metric: train_metrics[metric] + metric_value.item() for
                                 metric, metric_value
                                 in _train_metrics.items()}
                for metric, metric_value in _train_metrics.items():
                    self.tensorboard_logger.add_scalar(metric, metric_value.item())

                for feat_name in inputs:
                    total_padding = torch.sum(
                        (torch.ones_like(inputs[feat_name][1]) * inputs[feat_name][1][0]) - inputs[feat_name][1])
                    self.tensorboard_logger.add_scalar('total_padding_{}'.format(feat_name), total_padding.item())

                pbar.set_description('T e:{} l: {:.4f} a: {:.3f}'.format(epoch,
                                                                         loss["loss_final"].item(),
                                                                         _train_metrics['acc_lab_cd'].item()))
                pbar.update()
                #### /Logging ####

        log = {'train_loss': train_loss / len(data_loader),
               'train_metrics':
                   {metric: train_metrics[metric] / len(data_loader)
                    for metric in train_metrics}}
        global_step = epoch * len(data_loader)
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

        dataset = KaldiDataset(self.config['datasets'][valid_data]['features'],
                               self.config['datasets'][valid_data]['labels'],
                               self.model.context_left, self.model.context_right,
                               self.config['training']['max_seq_length_valid'],
                               self.tensorboard_logger,
                               self.debug, self.local)

        valid_data_loader = KaldiDataLoader(dataset,
                                            self.config['training']['batch_size_valid'],
                                            use_gpu=self.config["exp"]["n_gpu"] > 0,
                                            prefetch_to_gpu=self.config['exp']['prefetch_to_gpu'] is not None,
                                            device=self.device,
                                            num_workers=0)

        with tqdm(total=len(valid_data_loader), disable=not logger.isEnabledFor(logging.INFO)) as pbar:
            pbar.set_description('V e:{} l: {} '.format(epoch, '-'))
            for batch_idx, (sample_names, inputs, targets) in enumerate(valid_data_loader):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}

                output = self.model(inputs)
                loss = self.loss(output, targets)

                #### Logging ####
                valid_loss += loss["loss_final"].item()
                _eval_metrics = self._eval_metrics(output, targets)
                valid_metrics = {metric: valid_metrics[metric] + metric_value.item() for
                                 metric, metric_value
                                 in _eval_metrics.items()}
                pbar.set_description('V e:{} l: {:.4f} '.format(epoch, loss["loss_final"].item()))
                pbar.update()
                #### /Logging ####

        self.tensorboard_logger.set_step(global_step, 'valid')
        self.tensorboard_logger.add_scalar('valid_loss', valid_loss / len(valid_data_loader))
        for metric in valid_metrics:
            self.tensorboard_logger.add_scalar(metric, valid_metrics[metric] / len(valid_data_loader))

        return {'valid_loss': valid_loss / len(valid_data_loader),
                'valid_metrics': {metric: valid_metrics[metric] / len(valid_data_loader) for metric in
                                  valid_metrics}}

    def _eval_epoch(self, epoch, global_step):
        self.model.eval()
        batch_size = 1
        max_seq_length = -1

        test_data = self.config['data_use']['test_with']
        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        test_metrics = {metric: 0 for metric in self.metrics}

        dataset = KaldiDataset(self.config['datasets'][test_data]['features'],
                               self.config['datasets'][test_data]['labels'],
                               self.model.context_left, self.model.context_right,
                               max_sequence_length=max_seq_length,
                               tensorboard_logger=self.tensorboard_logger,
                               debug=self.debug)

        test_data_loader = KaldiDataLoader(dataset,
                                           batch_size,
                                           use_gpu=self.config["exp"]["n_gpu"] > 0,
                                           prefetch_to_gpu=self.config['exp']['prefetch_to_gpu'] is not None,
                                           device=self.device,
                                           num_workers=0)

        # paths of the output files (info,model,chunk_specific cfg file)
        base_file_name = '{}/exp_files/forward_{}_ep{:03d}'.format(out_folder, test_data, epoch)
        post_file = {}
        for out_name in self.config['test'].keys():
            if self.config['test'][out_name]['require_decoding']:
                out_file = '{}_{}_to_decode.ark'.format(base_file_name, out_name)
            else:
                out_file = '{}_{}.ark'.format(base_file_name, out_name)

            post_file[out_name] = kaldi_io.open_or_fd(out_file, 'wb')

        if self.config['exp']['prefetch_to_gpu'] is not None:
            test_data_loader.dataset.move_to(self.device)

        with tqdm(total=len(test_data_loader), disable=not logger.isEnabledFor(logging.INFO)) as pbar:
            pbar.set_description('E e:{}    '.format(epoch))
            for batch_idx, (sample_names, inputs, targets) in tqdm(enumerate(test_data_loader)):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}

                output = self.model(inputs)

                warned_label = False
                for output_label in output:
                    if output_label in self.config['test'].keys():

                        out_save = output[output_label].data.cpu().numpy()

                        if output_label in self.config['test'] and \
                                self.config['test'][output_label]['normalize_posteriors']:
                            # read the config file
                            counts = load_counts(self.config['test'][output_label]['normalize_with_counts_from_file'])
                            out_save = out_save - np.log(counts / np.sum(counts))

                            # save the output
                            # data_name = file ids
                            # out save shape <class 'tuple'>: (124, 1944)
                            # post_file dict out_dnn2: buffered wirter
                        assert len(out_save.shape) == 2
                        assert len(sample_names) == 1
                        kaldi_io.write_mat(post_file[output_label], out_save.squeeze(), sample_names[0])

                        #### Logging ####
                        _test_metrics = self._eval_metrics(output, targets)
                        test_metrics = {metric: test_metrics[metric] + metric_value.item() for
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
            self.tensorboard_logger.add_scalar(metric, test_metrics[metric] / len(test_data_loader))

        for out_name in self.config['test'].keys():
            post_file[out_name].close()

        test_metrics = {metric: test_metrics[metric] / len(test_data_loader)
                        for metric in test_metrics}

        decoding_results = []
        #### DECODING ####
        for out_lab in self.config['test']:

            # forward_data_lst = self.config['data_use']['test_with'] #TODO multiple forward sets
            forward_data_lst = [self.config['data_use']['test_with']]
            # forward_dec_outs = self.config['test'][out_lab]['require_decoding']

            for data in forward_data_lst:

                logger.debug('Decoding {} output {}'.format(data, out_lab))

                info_file = '{}/exp_files/decoding_{}_{}.info'.format(out_folder, data, out_lab)

                # create decode config file
                config_dec_file = '{}/decoding_{}_{}.conf'.format(out_folder, data, out_lab)
                config_dec = configparser.ConfigParser()
                config_dec.add_section('decoding')

                for dec_key in self.config['decoding'].keys():
                    config_dec.set('decoding', dec_key, str(self.config['decoding'][dec_key]))

                # add graph_dir, datadir, alidir
                lab_field = self.config['datasets'][data]['labels']['lab_cd']
                config_dec.set('decoding', 'alidir', os.path.abspath(lab_field['label_folder']))
                config_dec.set('decoding', 'data', os.path.abspath(lab_field['lab_data_folder']))
                config_dec.set('decoding', 'graphdir', os.path.abspath(lab_field['lab_graph']))

                with open(config_dec_file, 'w') as configfile:
                    config_dec.write(configfile)

                out_folder = os.path.abspath(out_folder)
                files_dec = '{}/exp_files/forward_{}_ep*_{}_to_decode.ark'.format(out_folder, data, out_lab)
                out_dec_folder = '{}/decode_{}_{}'.format(out_folder, data, out_lab)

                if not (os.path.exists(info_file)):
                    # Run the decoder
                    cmd_decode = '{}/{} {} {} \"{}\"'.format(
                        self.config['decoding']['decoding_script_folder'],
                        self.config['decoding']['decoding_script'],
                        os.path.abspath(config_dec_file),
                        out_dec_folder,
                        files_dec)
                    run_shell(cmd_decode)

                    # TODO remove ark files if needed
                    # if not forward_save_files:
                    #     list_rem = glob.glob(files_dec)
                    #     for rem_ark in list_rem:
                    #         os.remove(rem_ark)

                # Print WER results and write info file
                cmd_res = './scripts/check_res_dec.sh ' + out_dec_folder
                results = run_shell(cmd_res).decode('utf-8')
                logger.info(results)

                results = results.split("|")
                wer = float(results[0].split(" ")[1].strip())

                _corr, _sub, _del, _ins, _err, _s_err = [float(elem.strip())
                                                         for elem in results[2].split(" ") if len(elem) > 0]
                decoding_str = "WER: {} Corr: {} Sub: {} Del: {} Ins: {} Err: {}" \
                    .format(wer, _corr, _sub, _del, _ins, _err)
                logger.info(decoding_str)
                decoding_results.append(decoding_str)

                self.tensorboard_logger.add_text("WER results", decoding_str)

            # TODO plotting curves

        return {'test_metrics': test_metrics, "decoding_results": decoding_results}
