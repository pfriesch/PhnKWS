import os
import json
import threading
from multiprocessing import Queue

import torch

from base.utils import resume_checkpoint, save_checkpoint
from data import kaldi_io
from nn_.registries.lr_scheduler_registry import ReduceLROnPlateau
from utils.logger_config import logger
from utils.nvidia_smi import nvidia_smi_enabled, get_gpu_usage, get_gpu_memory_consumption
from utils.tensorboard_logger import WriterTensorboardX
from utils.util import ensure_dir, every, Timer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizers, lr_schedulers, seq_len_scheduler, restart_optim,
                 resume_path, config):
        self.config = config

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['exp']['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            raise NotImplementedError
            # self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.seq_len_scheduler = seq_len_scheduler

        self.epochs = config['exp']['n_epochs_tr']
        self.save_period = config['exp']['save_period']

        # self.monitor = config['exp'].get('monitor', 'off')

        # TODO  configuration to monitor model performance and save best
        # TODO  early stopping
        # configuration to monitor model performance and save best
        # if self.monitor == 'off':
        #     self.mnt_mode = 'off'
        #     self.mnt_best = 0
        # else:
        #     self.mnt_mode, self.mnt_metric = self.monitor.split()
        #     assert self.mnt_mode in ['min', 'max']
        #
        #     self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
        #     self.early_stop = config['exp'].get('early_stop', math.inf)

        self.start_epoch = 0
        self.global_step = 0

        self.out_dir = os.path.join(config['exp']['save_dir'], config['exp']['name'])

        # setup directory for checkpoint saving
        self.checkpoint_dir = os.path.join(self.out_dir, 'checkpoints')
        # setup visualization writer instance

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.out_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4, sort_keys=False)

        if resume_path:
            raise NotImplementedError
            # self.start_epoch, self.global_step, self.starting_dataset_sampler_state = \
            #     resume_checkpoint(resume_path, model, logger, optimizers, lr_schedulers)
            # for lr_scheduler in self.lr_schedulers:
            #     if self.lr_schedulers[lr_scheduler].current_lr() > self.config['training']['optimizer']['args']['lr']:
            #         self.lr_schedulers[lr_scheduler].set_lr(self.config['training']['optimizer']['args']['lr'])
            #
            # self.tensorboard_logger = WriterTensorboardX(os.path.join(self.out_dir, "tensorboard_logs"),
            #                                              purge_step=self.global_step)

        elif len(os.listdir(self.checkpoint_dir)) > 0:
            if not restart_optim:
                logger.info("Restarting training!")
                self.start_epoch, self.global_step, self.starting_dataset_sampler_state = \
                    resume_checkpoint(self.checkpoint_dir, model, logger, optimizers, lr_schedulers)
            else:
                logger.info("Warm starting training with new initialized optimizer!")
                self.start_epoch, self.global_step, self.starting_dataset_sampler_state = \
                    resume_checkpoint(self.checkpoint_dir, model, logger)
            # for lr_scheduler in self.lr_schedulers:
            #     if self.lr_schedulers[lr_scheduler].current_lr() > self.config['training']['optimizer']['args']['lr']:
            #         self.lr_schedulers[lr_scheduler].set_lr(self.config['training']['optimizer']['args']['lr'])

            self.tensorboard_logger = WriterTensorboardX(os.path.join(self.out_dir, "tensorboard_logs"),
                                                         purge_step=self.global_step)
        else:
            self.starting_dataset_sampler_state = None

            self.tensorboard_logger = WriterTensorboardX(os.path.join(self.out_dir, "tensorboard_logs"))

            # if hasattr(model, "get_sample_input"):
            #     self.tensorboard_logger.add_graph(model, model.get_sample_input(), True)

        self.device = 'cpu'
        if nvidia_smi_enabled:
            self.device = 'cuda:0'

            self.log_gpu_usage()

            self.stop_gpu_usage_logging = threading.Event()

            threading.Thread(target=lambda: every(30, self.log_gpu_usage, logger, self.stop_gpu_usage_logging)).start()

    def log_gpu_usage(self):
        if nvidia_smi_enabled:
            self.tensorboard_logger.add_scalar('usage', get_gpu_usage(), mode="gpu", global_step=self.global_step)
            self.tensorboard_logger.add_scalar('memory_usage_MiB', get_gpu_memory_consumption(), mode="gpu",
                                               global_step=self.global_step)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for self.epoch in range(self.start_epoch, self.epochs):
            logger.info('----- Epoch {} / {} -----'.format(format(self.epoch, "03d"), format(self.epochs, "03d")))
            logger.info(
                "Current lr: " + " ".join([f"{lr_scheduler_name}: {self.lr_schedulers[lr_scheduler_name].get_lr()}"
                                           for lr_scheduler_name in self.lr_schedulers]))
            logger.info(
                f"Current max seq len: {type(self.seq_len_scheduler).__name__}:"
                + f" {self.seq_len_scheduler.get_seq_len(self.epoch)}")

            self.tensorboard_logger.set_step(self.global_step, 'epoch_info')
            with Timer("elapsed_time_epoch", [self.tensorboard_logger, logger], self.global_step) as t:

                result_log = self._train_epoch(self.epoch)

            for lr_scheduler_name in self.lr_schedulers:
                if isinstance(self.lr_schedulers[lr_scheduler_name].get_lr(), float):
                    _curr_lr = self.lr_schedulers[lr_scheduler_name].get_lr()
                else:
                    _curr_lr = self.lr_schedulers[lr_scheduler_name].get_lr()[0]

                self.tensorboard_logger.add_scalar("lr_{}".format(lr_scheduler_name),
                                                   _curr_lr)
                if isinstance(self.lr_schedulers[lr_scheduler_name], ReduceLROnPlateau):
                    self.lr_schedulers[lr_scheduler_name].step(result_log['valid_loss'], epoch=self.epoch)
                else:
                    self.lr_schedulers[lr_scheduler_name].step(epoch=self.epoch)

                self.seq_len_scheduler.step(self.epoch)

            # save logged informations into log dict
            log = {'epoch': self.epoch}
            log.update({'elapsed_time_epoch': f"{t.interval / 60:.2f} min"})
            log.update(result_log)

            # print logged informations to the screen
            if logger is not None:
                for key, value in log.items():
                    logger.info('    {:10s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            # if self.mnt_mode != 'off':
            #     # check whether model performance improved or not, according to specified metric(mnt_metric)
            #     assert self.mnt_metric in log, \
            #         f"Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled."
            #     improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
            #                (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
            #
            #     if improved:
            #         self.mnt_best = log[self.mnt_metric]
            #         not_improved_count = 0
            #         best = True
            #     else:
            #         not_improved_count += 1
            #
            #     if not_improved_count > self.early_stop:
            #         logger.info(f"Validation performance didn\'t improve for {self.early_stop} epochs. Training stops.")
            #         break

            if self.epoch % self.save_period == 0:
                save_checkpoint(self.epoch, self.global_step, self.model, self.optimizers, self.lr_schedulers,
                                self.seq_len_scheduler, self.config, self.checkpoint_dir,
                                # self.monitor_best,
                                save_best=best)

        # result_eval = self._eval_epoch(epoch)
        # logger.info(result_eval)
        if nvidia_smi_enabled:
            self.stop_gpu_usage_logging.set()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval_epoch(self, epoch):
        raise NotImplementedError


def to_device(device, data, non_blocking=True):
    if isinstance(data, dict):
        return {k: to_device(device, v) for k, v in data.items()}
    if isinstance(data, list):
        return [e.to(device, non_blocking=non_blocking) for e in data]
    else:
        return data.to(device, non_blocking=non_blocking)


def detach(data):
    if isinstance(data, dict):
        return {k: v.detach() for k, v in data.items()}
    if isinstance(data, list):
        return [e.detach() for e in data]
    else:
        return data.detach()


def detach_cpu(data):
    if isinstance(data, dict):
        return {k: v.detach().cpu() for k, v in data.items()}
    if isinstance(data, list):
        return [e.detach().cpu() for e in data]
    else:
        return data.detach().cpu()


def eval_metrics(output_target, metrics):
    output, target = output_target
    acc_metrics = {}
    for metric in metrics:
        acc_metrics[metric] = metrics[metric](output, target)
    return acc_metrics


def detach_metrics(output, targets):
    detached_output = {}
    for k in output:
        detached_output[k] = output[k].detach().cpu()
    detached_targets = {}
    for k in targets:
        detached_targets[k] = targets[k].detach().cpu()
    return detached_output, detached_targets


def metrics_accumulator(q: Queue, metrics):
    _accumulated_valid_metrics = {metric: 0 for metric in metrics}
    while True:
        metrics_input = q.get(True)
        if metrics_input is None:
            break
        _output, _targets = metrics_input
        _valid_metrics = eval_metrics((_output, _targets), metrics)
        for metric, metric_value in _valid_metrics.items():
            _accumulated_valid_metrics[metric] += metric_value
    return _accumulated_valid_metrics


class KaldiOutputWriter:

    def __init__(self, out_folder, data_name, output_names, epoch):
        super().__init__()
        self.out_folder = out_folder
        self.data_name = data_name
        self.epoch = epoch
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
        """

        :param out_name:
        :param out_save: shape should be [L, C] / [length, channels] for kaldi fst decoder
        :param sample_name:
        :return:
        """
        kaldi_io.write_mat(self.post_file[out_name], out_save, sample_name)
