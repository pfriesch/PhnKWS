from torch import optim


def optimizer_init(config, model):
    if config['training']['optimizer']['type'] == 'adam':
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizers = [optim.Adam(trainable_params, **config['training']['optimizer']["args"])]

    elif config['training']['optimizer']['type'] == 'triple_rmsprop':
        optimizers = []
        trainable_params_lstm = filter(lambda p: p.requires_grad, model.lstm.parameters())
        optimizers.append(optim.RMSprop(trainable_params_lstm,
                                        lr=0.0016,
                                        alpha=0.95,
                                        eps=1e-8,
                                        weight_decay=0.0,
                                        momentum=0,
                                        centered=False))

        trainable_params_mlp_cd = filter(lambda p: p.requires_grad, model.mlp_lab_cd.parameters())
        optimizers.append(optim.RMSprop(trainable_params_mlp_cd,
                                        lr=0.0016,
                                        alpha=0.95,
                                        eps=1e-8,
                                        weight_decay=0.0,
                                        momentum=0,
                                        centered=False))

        trainable_params_mlp_mono = filter(lambda p: p.requires_grad, model.mlp_lab_cd.parameters())
        optimizers.append(optim.RMSprop(trainable_params_mlp_mono,
                                        lr=0.0004,
                                        alpha=0.95,
                                        eps=1e-8,
                                        weight_decay=0.0,
                                        momentum=0,
                                        centered=False))


    else:
        raise ValueError

    return optimizers