from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, _LRScheduler

from utils.logger_config import logger


class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logger.info('Epoch {:d}: reducing learning rate'
                                ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    # def set_lr(self, new_lr):
    #     for i, param_group in enumerate(self.optimizer.param_groups):
    #         param_group['lr'] = new_lr
    #         if self.verbose:
    #             logger.info(f'Setting learning rate of group {i} to {new_lr:.4e}.')


class StaticLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(StaticLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


def lr_scheduler_init(config, optimizers):
    lr_schedulers = {}
    if config['training']['lr_scheduler']['name'] == 'ReduceLROnPlateau':

        for opti_name in optimizers:
            input_dict = {"mode": 'min',
                          "factor": 0.4,
                          "patience": 2,
                          "verbose": True,
                          "cooldown": 2,
                          "threshold_mode": 'rel'}
            input_dict.update(config['training']['lr_scheduler']['args'])
            lr_schedulers[opti_name] = ReduceLROnPlateau(optimizers[opti_name], **input_dict)

    elif config['training']['lr_scheduler']['name'] == 'MultiStepLR':

        for opti_name in optimizers:
            lr_schedulers[opti_name] = MultiStepLR(optimizers[opti_name],
                                                   **config['training']['lr_scheduler']['args'])

    elif config['training']['lr_scheduler']['name'] == 'ExponentialLR':

        for opti_name in optimizers:
            lr_schedulers[opti_name] = ExponentialLR(optimizers[opti_name],
                                                     **config['training']['lr_scheduler']['args'])

    elif config['training']['lr_scheduler']['name'] == 'StaticLR':
        for opti_name in optimizers:
            lr_schedulers[opti_name] = StaticLR(optimizers[opti_name],
                                                **config['training']['lr_scheduler']['args'])

    else:
        raise ValueError

    return lr_schedulers
