from torch import optim


def optimizer_init(config, model):
    optimizers = {}
    if config['training']['optimizer']['type'] == 'CE_triple_rmsprop_cfg':
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

    elif config['training']['optimizer']['type'] == 'sgd':
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizers['all'] = optim.SGD(trainable_params, **config['training']['optimizer']["args"])
    elif config['training']['optimizer']['type'] == 'adam':
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizers['all'] = optim.Adam(trainable_params, **config['training']['optimizer']["args"])

    else:
        raise ValueError(f"Can't find the optimizer {config['training']['optimizer']['type']}")

    return optimizers
