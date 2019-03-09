import os

import numpy as np

from base.base_trainer import BaseTrainer
from nn_.losses.ctc_phn import CTCPhnLoss
from trainer.eval import evaluate
from trainer.train_epoch import train_epoch
from trainer.valid import valid_epoch_sync_metrics, valid_epoch_async_metrics


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizers, lr_schedulers, seq_len_scheduler,
                 resume_path, config,
                 restart_optim,
                 do_validation, overfit_small_batch):
        super(Trainer, self).__init__(model, loss, metrics, optimizers, lr_schedulers,
                                      seq_len_scheduler, restart_optim, resume_path, config)
        self.config = config
        self.do_validation = do_validation
        self.log_step = int(np.sqrt(config['training']['batching']['batch_size_train']))
        self.overfit_small_batch = overfit_small_batch
        # Necessary for cudnn ctc function
        self.max_label_length = 256 if isinstance(self.loss, CTCPhnLoss) else None

    def _train_epoch(self, epoch):
        log, self.tarting_dataset_sampler_state, self.global_step \
            = train_epoch(epoch, self.global_step, self.model,
                          self.loss, self.metrics, self.config,
                          self.max_label_length, self.device,
                          self.tensorboard_logger,
                          self.seq_len_scheduler,
                          self.overfit_small_batch,
                          self.starting_dataset_sampler_state,
                          self.optimizers,
                          self.lr_schedulers, self.do_validation,
                          self._valid_epoch, self.checkpoint_dir)
        return log

    def _valid_epoch(self, epoch):
        """
       Validate after training an epoch
       :return: A log that contains information about validation
       Note:
           The validation metrics in log must have the key 'val_metrics'.
       """
        if 'DEBUG_MODE' in os.environ and bool(int(os.environ['DEBUG_MODE'])):
            return valid_epoch_sync_metrics(epoch, self.model, self.loss, self.metrics, self.config,
                                            self.max_label_length, self.device, self.tensorboard_logger)
        else:
            return valid_epoch_async_metrics(epoch, self.model, self.loss, self.metrics, self.config,
                                             self.max_label_length, self.device, self.tensorboard_logger)

    def _eval_epoch(self, epoch):
        test_data = self.config['dataset']['data_use']['test_with']
        return evaluate(self.model,
                        metrics=self.metrics,
                        device=self.device,
                        out_folder=os.path.join(self.config['exp']['save_dir'], self.config['exp']['name']),
                        exp_name=self.config['exp']['name'],
                        max_label_length=self.max_label_length,
                        epoch=epoch,
                        dataset_type=self.config['training']['dataset_type'],
                        data_cache_root=self.config['exp']['data_cache_root'],
                        test_with=test_data,
                        all_feats_dict=self.config['dataset']['dataset_definition']['datasets'][test_data]['features'],
                        features_use=self.config['dataset']['features_use'],
                        all_labs_dict=self.config['dataset']['dataset_definition']['datasets'][test_data]['labels'],
                        labels_use=self.config['dataset']['labels_use'],
                        phoneme_dict=self.config['dataset']['dataset_definition']['phoneme_dict'])
