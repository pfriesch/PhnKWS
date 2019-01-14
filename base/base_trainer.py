import os
import math
import json
import time

import torch

from utils.logger_config import logger
from utils.tensorboard_logger import WriterTensorboardX
from utils.util import ensure_dir, folder_to_checkpoint


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizers, lr_schedulers, resume_path, config):
        self.config = config

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

        # setup directory for checkpoint saving
        self.checkpoint_dir = os.path.join(config['exp']['save_dir'], config['exp']['name'], 'checkpoints')
        # setup visualization writer instance
        self.tensorboard_logger = WriterTensorboardX(
            os.path.join(config['exp']['save_dir'], config['exp']['name'], "logs"))

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(config['exp']['save_dir'], config['exp']['name'], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume_path:
            self._resume_checkpoint(resume_path)

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

            start_time = time.time()
            result = self._train_epoch(epoch)
            elapsed_time_epoch = time.time() - start_time
            self.tensorboard_logger.add_scalar("elapsed_time_epoch", elapsed_time_epoch)

            for _idx, lr_scheduler in enumerate(self.lr_schedulers):
                self.tensorboard_logger.add_scalar("lr_{}".format(_idx), lr_scheduler.current_lr())
                lr_scheduler.step(result['valid_loss'], epoch=epoch)

            self.tensorboard_logger.add_scalar("max_seq_length_train_curr", self.max_seq_length_train_curr)
            if self.config['training']['increase_seq_length_train']:
                self.max_seq_length_train_curr *= self.config['training']['multply_factor_seq_len_train']
                if self.max_seq_length_train_curr > self.config['training']['max_seq_length_train']:
                    self.max_seq_length_train_curr = self.config['training']['max_seq_length_train']

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update({'epoch_elapsed_time': elapsed_time_epoch})
            log.update(result)

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
                self._save_checkpoint(epoch, save_best=best)

        result_eval = self._eval_epoch(epoch)
        logger.info(result_eval)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizers': [opti.state_dict() for opti in self.optimizers],
            'lr_schedulers': [lr_scheduler.state_dict() for lr_scheduler in self.lr_schedulers],
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

    def _resume_checkpoint(self, resume_folder_path):
        resume_path = folder_to_checkpoint(resume_folder_path)

        logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']: #TODO add check
        #     logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
        #                         'Optimizer parameters not being resumed.')
        # else:
        for _idx, opti in enumerate(checkpoint['optimizer']):
            self.optimizers[_idx].load_state_dict(opti)
        for _idx, lr_scheduler in enumerate(checkpoint['lr_scheduler']):
            self.lr_schedulers[_idx].load_state_dict(lr_scheduler)

        logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
