import os
import math
import json
import logging
import datetime
import time
import numpy as np

import torch
from utils.util import ensure_dir
from utils.utils import get_posterior_norm_data
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config, logger=None):
        self.config = config
        self.logger = logger

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['exp']['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

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
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        # self.checkpoint_dir = os.path.join(config['exp']['save_dir'], config['exp']['name'], start_time) #TODO switch to make folder by time
        self.checkpoint_dir = os.path.join(config['exp']['save_dir'], config['exp']['name'])
        # setup visualization writer instance
        writer_dir = os.path.join(self.checkpoint_dir, "logs")
        self.writer = WriterTensorboardX(writer_dir, enable=True)

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
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
            print('--------------------- Epoch %s / %s ---------------------' % (
                format(epoch, "03d"), format(self.epochs, "03d")))
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            if self.logger is not None:
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
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
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        result_eval = self._eval_epoch(epoch)
        print(result_eval)

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
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
