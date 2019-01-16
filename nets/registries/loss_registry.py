from nets.losses.cd_loss import CDLoss
from nets.losses.mtl_mono_cd_loss import MtlMonoCDLoss


def loss_init(config):
    if config['arch']['loss']['name'] == 'mtl_mono_cd':
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    elif config['arch']['loss']['name'] == 'lab_cd':
        return CDLoss()

    else:
        raise ValueError
