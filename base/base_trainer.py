import os
import math
import json
import threading

import torch

from base.utils import resume_checkpoint
from utils.logger_config import logger
from utils.nvidia_smi import nvidia_smi_enabled, get_gpu_usage, get_gpu_memory_consumption
from utils.tensorboard_logger import WriterTensorboardX
from utils.util import ensure_dir, every, Timer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizers, lr_schedulers, decoding_norm_data, resume_path, config):
        self.config = config
        self.decoding_norm_data = decoding_norm_data

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['exp']['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.max_seq_length_train_curr = self.config['training']['start_seq_len_train']

        self.epochs = config['exp']['n_epochs_tr']
        self.save_period = config['exp']['save_period']
        self.monitor = config['exp'].get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = config['exp'].get('early_stop', math.inf)

        self.start_epoch = 0
        self.global_step = 0

        self.out_dir = os.path.join(config['exp']['save_dir'], config['exp']['name'])

        # setup directory for checkpoint saving
        self.checkpoint_dir = os.path.join(self.out_dir, 'checkpoints')
        # setup visualization writer instance
        self.tensorboard_logger = WriterTensorboardX(
            os.path.join(self.out_dir, "logs"))

        # if hasattr(model, "get_sample_input"):
        #     self.tensorboard_logger.add_graph(model, model.get_sample_input(), True)

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.out_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4, sort_keys=False)

        if resume_path:
            self.start_epoch, self.global_step, decoding_norm_data = \
                resume_checkpoint(resume_path, model, logger, optimizers, lr_schedulers)
            assert decoding_norm_data == self.decoding_norm_data

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

        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            logger.info('----- Epoch {} / {} -----'.format(format(epoch, "03d"), format(self.epochs, "03d")))

            self.tensorboard_logger.set_step(self.global_step, 'epoch_info')
            with Timer("elapsed_time_epoch", [self.tensorboard_logger, logger], self.global_step) as t:
                result_log = self._train_epoch(epoch)

            for lr_scheduler_name in self.lr_schedulers:
                self.tensorboard_logger.add_scalar("lr_{}".format(lr_scheduler_name),
                                                   self.lr_schedulers[lr_scheduler_name].current_lr())
                self.lr_schedulers[lr_scheduler_name].step(result_log['valid_loss'], epoch=epoch)

            if self.max_seq_length_train_curr is not None:
                self.tensorboard_logger.add_scalar("max_seq_length_train_curr", self.max_seq_length_train_curr)
                if self.config['training']['increase_seq_length_train']:
                    self.max_seq_length_train_curr *= self.config['training']['multply_factor_seq_len_train']
                    if self.max_seq_length_train_curr > self.config['training']['max_seq_length_train']:
                        self.max_seq_length_train_curr = self.config['training']['max_seq_length_train']

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update({'elapsed_time_epoch': t.interval})
            log.update(result_log)

            # print logged informations to the screen
            if logger is not None:
                for key, value in log.items():
                    logger.info('    {:10s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    logger.info(
                        "Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, save_best=best)

        result_eval = self._eval_epoch(epoch)
        logger.info(result_eval)
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

    def save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizers': {opti_name: self.optimizers[opti_name].state_dict() for opti_name in self.optimizers},
            'lr_schedulers': {lr_sched_name: self.lr_schedulers[lr_sched_name].state_dict()
                              for lr_sched_name in self.lr_schedulers},
            'decoding_norm_data': self.decoding_norm_data,
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))

        if epoch > 3:
            filename_prev = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch - 3))
            if os.path.exists(filename_prev):
                os.remove(filename_prev)
                logger.info("Removing old checkpoint: {} ...".format(filename_prev))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            logger.info("Saving current best: {} ...".format('model_best.pth'))
