from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from utils.logger_config import logger


class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):

    def current_lr(self):
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

    def set_lr(self, new_lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = new_lr
            if self.verbose:
                logger.info(f'Setting learning rate of group {i} to {new_lr:.4e}.')


def lr_scheduler_init(config, optimizers):
    lr_schedulers = {}
    if config['training']['lr_scheduler']['name'] == 'ReduceLROnPlateau':

        for opti_name in optimizers:
            lr_schedulers[opti_name] = ReduceLROnPlateau(optimizers[opti_name],
                                                         mode='min',

                                                         # factor=config['training']['lr_scheduler']['arch_halving_factor'],
                                                         # patience=config['training']['lr_scheduler']['patience'],
                                                         # cooldown=config['training']['lr_scheduler']['cooldown'],
                                                         # min_lr=config['training']['lr_scheduler']['min_lr'],
                                                         verbose=True,
                                                         # threshold=config['training']['lr_scheduler'][
                                                         #     'arch_improvement_threshold'],
                                                         threshold_mode='rel',
                                                         **config['training']['lr_scheduler']['args'])

    elif config['training']['lr_scheduler']['name'] == 'MultiStepLR':

        for opti_name in optimizers:
            lr_schedulers[opti_name] = MultiStepLR(optimizers[opti_name],
                                                   **config['training']['lr_scheduler']['args'])
    else:
        raise ValueError

    return lr_schedulers
