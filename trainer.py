import configparser
import glob
import json
import logging
import os
import re

import numpy as np
from tqdm import tqdm

import kaldi_io
from base.base_trainer import BaseTrainer
from data_loader.kaldi_data_loader import KaldiDataLoader
from data_loader.kaldi_dataset import load_counts
from utils.utils import compute_n_chunks, \
    check_environment, get_chunk_config, run_shell

check_environment()


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizer, resume, config, do_validation,
                 lr_scheduler=None, logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, logger)
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.do_validation = do_validation
        self.log_step = int(np.sqrt(config['batches']['batch_size_train']))

        if self.config['batches']['increase_seq_length_train']:
            self.max_seq_length_curr = self.config['batches']['start_seq_len_train']
        else:
            self.max_seq_length_curr = self.config['batches']['max_seq_length_train']

    def _eval_metrics(self, output, target):
        acc_metrics = {}
        for metric in self.metrics:
            acc_metrics[metric] = self.metrics[metric](output, target)
            self.writer.add_scalar(metric, acc_metrics[metric])
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
        tr_data = 'TIMIT_tr'
        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        N_ck_tr = compute_n_chunks(out_folder, tr_data, epoch, 'train')

        total_loss = 0
        total_metrics = {metric: 0 for metric in self.metrics}
        total_len = 0

        for chunk in range(N_ck_tr):

            # path of the list of features for this chunk
            lst_file = '{}/exp_files/train_{}_ep{:03d}_ck{:02d}_*.lst' \
                .format(out_folder, tr_data, epoch, chunk)

            # paths of the output files (info,model,chunk_specific cfg file)
            info_file_name = '{}/exp_files/train_{}_ep{:03d}_ck{:02d}.info' \
                .format(out_folder, tr_data, epoch, chunk)

            if not os.path.exists(info_file_name):

                chunk_config = get_chunk_config(self.config, lst_file, info_file_name, "train", tr_data,
                                                self.max_seq_length_curr, epoch, chunk)
                # TODO add info file after run and if done skip step

                data_loader = KaldiDataLoader(chunk_config['data_chunk']['fea'],
                                              chunk_config['data_chunk']['lab'],
                                              chunk_config['batches']['batch_size_train'],
                                              shuffle=False,
                                              use_gpu=self.config["exp"]["n_gpu"] > 0,
                                              prefetch_to_gpu=self.config['exp']['prefetch_to_gpu'],
                                              device=self.device,
                                              num_workers=0)

                for batch_idx, (sample_names, inputs, targets) in tqdm(enumerate(data_loader)):

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    targets = {k: v.to(self.device) for k, v in targets.items()}

                    self.optimizer.zero_grad()
                    output = self.model(inputs)
                    loss = self.loss(output, targets)
                    loss["loss_final"].backward()
                    self.optimizer.step()

                    self.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
                    for _loss in loss:
                        self.writer.add_scalar(_loss, loss[_loss].item())
                    total_loss += loss["loss_final"].item()
                    total_metrics = {metric: total_metrics[metric] + metric_value.item() for metric, metric_value in
                                     self._eval_metrics(output, targets).items()}
                    total_len += len(data_loader)

                    if self.logger.isEnabledFor(logging.INFO) and batch_idx % self.log_step == 0:
                        self.logger.info('Train Epoch: {} Chunk: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            epoch, chunk,
                            batch_idx * data_loader.batch_size,
                            data_loader.n_samples,
                            100.0 * batch_idx / len(data_loader),
                            loss["loss_final"].item()))
                # Write info file
                with open(info_file_name, "w") as json_file:
                    results = {}
                    results['total_loss'] = total_loss / total_len
                    results['total_metrics'] = {metric: total_metrics[metric] / total_len for metric in total_metrics}
                    # TODO log time for benchmarking
                    json.dump(results, json_file)
            else:
                raise NotImplementedError("read results")

        if total_len > 0:
            log = {
                'train_loss': total_loss / total_len,
                'train_metrics': {metric: total_metrics[metric] / total_len for metric in total_metrics}
            }
        else:
            log = {
                'train_loss': 0,
                'train_metrics': {metric: 0 for metric in total_metrics}
            }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            # TODO check if that is how it's done
            #
            #  computing average validation error (on all the dataset specified)
            # err_valid_mean = np.mean(np.asarray(list(valid_peformance_dict.values()))[:, 1])
            # err_valid_mean_prev = np.mean(np.asarray(list(valid_peformance_dict_prev.values()))[:, 1])
            #
            # for lr_arch in list(lr.keys()):
            #     if ((err_valid_mean_prev - err_valid_mean) / err_valid_mean) < improvement_threshold[lr_arch]:
            #         lr[lr_arch] = lr[lr_arch] * halving_factor[lr_arch]

            self.lr_scheduler.step()

        #  if needed, update sentence_length
        if self.config['batches']['increase_seq_length_train']:
            self.max_seq_length_curr = self.max_seq_length_curr * int(
                self.config['batches']['multply_factor_seq_len_train'])
            if self.max_seq_length_curr > self.config['batches']['max_seq_length_train']:
                self.max_seq_length_curr = self.config['batches']['max_seq_length_train']

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        valid_data = 'TIMIT_dev'
        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        N_ck_tr = compute_n_chunks(out_folder, valid_data, epoch, 'valid')

        total_val_loss = 0
        total_val_metrics = {metric: 0 for metric in self.metrics}
        total_val_len = 0

        for chunk in range(N_ck_tr):

            # path of the list of features for this chunk
            lst_file = '{}/exp_files/valid_{}_ep{:03d}_ck{:02d}_*.lst' \
                .format(out_folder, valid_data, epoch, chunk)

            # paths of the output files (info,model,chunk_specific cfg file)
            info_file_name = '{}/exp_files/valid_{}_ep{:03d}_ck{:02d}.info' \
                .format(out_folder, valid_data, epoch, chunk)
            if not os.path.exists(info_file_name):

                chunk_config = get_chunk_config(self.config, lst_file, info_file_name, "valid", valid_data,
                                                self.max_seq_length_curr, epoch, chunk)

                valid_data_loader = KaldiDataLoader(chunk_config['data_chunk']['fea'],
                                                    chunk_config['data_chunk']['lab'],
                                                    chunk_config['batches']['batch_size_valid'],
                                                    shuffle=False,
                                                    use_gpu=self.config["exp"]["n_gpu"] > 0,
                                                    prefetch_to_gpu=self.config['exp']['prefetch_to_gpu'] is not None,
                                                    device=self.device,
                                                    num_workers=0)

                for batch_idx, (sample_names, inputs, targets) in tqdm(enumerate(valid_data_loader)):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    targets = {k: v.to(self.device) for k, v in targets.items()}

                    output = self.model(inputs)
                    loss = self.loss(output, targets)

                    self.writer.set_step((epoch - 1) * len(valid_data_loader) + batch_idx, 'valid')
                    self.writer.add_scalar('loss', loss["loss_final"].item())
                    self.writer.add_scalar('loss_cd', loss["loss_cd"].item())
                    self.writer.add_scalar('loss_mono', loss["loss_mono"].item())
                    total_val_loss += loss["loss_final"].item()
                    total_val_metrics = {metric: total_val_metrics[metric] + metric_value for metric, metric_value in
                                         self._eval_metrics(output, targets).items()}
                    total_val_len += len(valid_data_loader)

                # Write info file
                with open(info_file_name, "w") as json_file:
                    results = {}
                    results['total_val_loss'] = total_val_loss / total_val_len
                    results['total_val_metrics'] = {metric: total_val_metrics[metric] / total_val_len for metric in
                                                    total_val_metrics}
                    # TODO log time for benchmarking
                    json.dump(results, json_file)

            else:
                raise NotImplementedError("read results")

        if total_val_len > 0:
            return {
                'val_loss': total_val_loss / total_val_len,
                'val_metrics': {metric: total_val_metrics[metric] / total_val_len for metric in total_val_metrics}
            }
        else:
            return {
                'val_loss': 0,
                'val_metrics': {metric: 0 for metric in total_val_metrics}
            }

    def _eval_epoch(self, epoch):
        self.model.eval()
        batch_size = 1
        max_seq_length = -1

        test_data = 'TIMIT_test'  # TODO change to test
        out_folder = os.path.join(self.config['exp']['save_dir'], self.config['exp']['name'])

        N_ck_tr = compute_n_chunks(out_folder, test_data, epoch, 'forward')

        total_test_loss = 0
        total_test_metrics = {metric: 0 for metric in self.metrics}
        total_test_len = 0

        for chunk in range(N_ck_tr):

            # path of the list of features for this chunk
            lst_file = '{}/exp_files/forward_{}_ep{:03d}_ck{:02d}_*.lst' \
                .format(out_folder, test_data, epoch, chunk)

            # paths of the output files (info,model,chunk_specific cfg file)
            info_file_name = '{}/exp_files/forward_{}_ep{:03d}_ck{:02d}.info' \
                .format(out_folder, test_data, epoch, chunk)

            chunk_config = get_chunk_config(self.config, lst_file, info_file_name, "forward", test_data,
                                            self.max_seq_length_curr, epoch, chunk)

            test_data_loader = KaldiDataLoader(chunk_config['data_chunk']['fea'],
                                               chunk_config['data_chunk']['lab'],
                                               batch_size,
                                               shuffle=False,
                                               use_gpu=self.config["exp"]["n_gpu"] > 0,
                                               prefetch_to_gpu=self.config['exp']['prefetch_to_gpu'] is not None,
                                               device=self.device,
                                               num_workers=0)

            post_file = {}
            for out_name in self.config['forward'].keys():
                if self.config['forward'][out_name]['require_decoding']:
                    out_file = info_file_name.replace('.info', '_' + out_name + '_to_decode.ark')
                else:
                    out_file = info_file_name.replace('.info', '_' + out_name + '.ark')

                post_file[out_name] = kaldi_io.open_or_fd(out_file, 'wb')

            if self.config['exp']['prefetch_to_gpu'] is not None:
                # start_time = time.time() #TODO benchmark everything
                test_data_loader.dataset.move_to(self.device)
                # elapsed_time_load = time.time() - start_time

            for batch_idx, (sample_names, inputs, targets) in tqdm(enumerate(test_data_loader)):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}

                output = self.model(inputs)

                for output_lab in output:
                    if output_lab in self.config['forward'].keys():

                        out_save = output[output_lab].data.cpu().numpy()

                        if output_lab in self.config['forward'] and self.config['forward'][output_lab][
                            'normalize_posteriors']:
                            # read the config file
                            counts = load_counts(self.config['forward'][output_lab]['normalize_with_counts_from'])
                            out_save = out_save - np.log(counts / np.sum(counts))

                            # save the output
                            # data_name = file ids
                            # out save shape <class 'tuple'>: (124, 1944)
                            # post_file dict out_dnn2: buffered wirter
                        assert out_save.shape[1] == 1
                        assert len(sample_names) == 1
                        kaldi_io.write_mat(post_file[output_lab], out_save.squeeze(), sample_names[0])

                        self.writer.set_step((epoch - 1) * len(test_data_loader) + batch_idx, 'forward')

                        total_test_metrics = {metric: total_test_metrics[metric] + metric_value
                                              for metric, metric_value in
                                              self._eval_metrics(output, targets).items()}
                        total_test_len += len(test_data_loader)
                    else:
                        self.logger.debug("Skipping saving forward for decoding for key {}".format(output_lab))

            for out_name in self.config['forward'].keys():
                post_file[out_name].close()

            #### DECODING ####

            # --------DECODING--------#
            # dec_lst = glob.glob(out_folder + '/exp_files/*_to_decode.ark')

            for out_lab in self.config['forward']:

                forward_data_lst = self.config['data_use']['forward_with']
                # forward_dec_outs = self.config['forward'][out_lab]['require_decoding']

                for data in forward_data_lst:

                    print('Decoding %s output %s' % (data, out_lab))

                    info_file = '{}/exp_files/decoding_{}_{}.info'.format(out_folder, data, out_lab)

                    # create decode config file
                    config_dec_file = '{}/decoding_{}_{}.conf'.format(out_folder, data, out_lab)
                    config_dec = configparser.ConfigParser()
                    config_dec.add_section('decoding')

                    for dec_key in self.config['decoding'].keys():
                        config_dec.set('decoding', dec_key, str(self.config['decoding'][dec_key]))

                    # add graph_dir, datadir, alidir
                    lab_field = self.config['datasets'][data]['lab']['lab_cd']
                    config_dec.set('decoding', 'alidir', os.path.abspath(lab_field['lab_folder']))
                    config_dec.set('decoding', 'data', os.path.abspath(lab_field['lab_data_folder']))
                    config_dec.set('decoding', 'graphdir', os.path.abspath(lab_field['lab_graph']))

                    with open(config_dec_file, 'w') as configfile:
                        config_dec.write(configfile)

                    out_folder = os.path.abspath(out_folder)
                    files_dec = '{}/exp_files/forward_{}_ep*_ck*_{}_to_decode.ark'.format(out_folder, data, out_lab)
                    out_dec_folder = '{}/decode_{}_{}'.format(out_folder, data, out_lab)

                    if not (os.path.exists(info_file)):
                        # Run the decoder
                        cmd_decode = '{}/{} {} {} \"{}\"'.format(
                            self.config['decoding']['decoding_script_folder'],
                            self.config['decoding']['decoding_script'],
                            os.path.abspath(config_dec_file),
                            out_dec_folder,
                            files_dec)
                        run_shell(cmd_decode, self.logger)

                        # TODO remove ark files if needed
                        # if not forward_save_files:
                        #     list_rem = glob.glob(files_dec)
                        #     for rem_ark in list_rem:
                        #         os.remove(rem_ark)

                    # Print WER results and write info file
                    cmd_res = './scripts/check_res_dec.sh ' + out_dec_folder
                    wers = run_shell(cmd_res, self.logger).decode('utf-8')
                    # res_file = open(res_file_path, "a") #TODO
                    # res_file.write('%s\n' % wers)
                    print(wers)

            # TODO Saving Loss and Err as .txt and plotting curves

            if total_test_len > 0:
                return {
                    'val_loss': total_test_loss / total_test_len,
                    'val_metrics': {metric: total_test_metrics[metric] / total_test_len for metric in
                                    total_test_metrics}
                }
            else:
                return {
                    'val_loss': 0,
                    'val_metrics': {metric: 0 for metric in total_test_metrics}
                }
