from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from nn_.registries.lr_scheduler_registry import lr_scheduler_init


class StaticLR(_LRScheduler):

    def __init__(self, optimizer):
        super(StaticLR, self).__init__(optimizer)

    def step(self, epoch=None):
        pass


def optimizer_init(config, model, optim_overwrite):
    optimizers = {}
    if not bool(optim_overwrite):
        optim_config = config['training']['optimizer']
    else:
        optim_config = optim_overwrite
    if optim_config['type'] == 'CE_triple_rmsprop_cfg':
        trainable_params_MLP = filter(lambda p: p.requires_grad, model.MLP.parameters())
        optimizers['MLP'] = optim.SGD(trainable_params_MLP,
                                      lr=0.08,
                                      weight_decay=0.0,
                                      momentum=0)

        trainable_params_linear_lab_cd = filter(lambda p: p.requires_grad, model.linear_lab_cd.parameters())
        optimizers['linear_lab_cd'] = optim.RMSprop(trainable_params_linear_lab_cd,
                                                    lr=0.0004,
                                                    alpha=0.95,
                                                    eps=1e-8,
                                                    weight_decay=0.0,
                                                    momentum=0,
                                                    centered=False)

        trainable_params_linear_lab_mono = filter(lambda p: p.requires_grad, model.linear_lab_mono.parameters())
        optimizers['linear_lab_mono'] = optim.RMSprop(trainable_params_linear_lab_mono,
                                                      lr=0.0004,
                                                      alpha=0.95,
                                                      eps=1e-8,
                                                      weight_decay=0.0,
                                                      momentum=0,
                                                      centered=False)

    elif optim_config['type'] == 'sgd':
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizers['all'] = optim.SGD(trainable_params, **optim_config["args"])
    elif optim_config['type'] == 'adam':
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizers['all'] = optim.Adam(trainable_params, **optim_config["args"])

    else:
        raise ValueError(f"Can't find the optimizer {optim_config['type']}")
    lr_schedulers = lr_scheduler_init(config, optimizers)

    return optimizers, lr_schedulers
