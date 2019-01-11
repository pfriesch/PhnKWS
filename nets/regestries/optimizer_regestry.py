from torch import optim


def optimizer_init(config, trainable_params):
    if config['training']['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(trainable_params, **config['training']['optimizer']["args"])
    else:
        raise ValueError

    return optimizer

