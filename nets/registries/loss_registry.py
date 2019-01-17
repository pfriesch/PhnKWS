from nets.losses.cd_loss import CDLoss
from nets.losses.ctc_mtl_mono_cd_loss import CTCMtlMonoCDLoss
from nets.losses.mtl_mono_cd_loss import MtlMonoCDLoss


def loss_init(config):
    if config['arch']['loss']['name'] == 'mtl_mono_cd':
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    elif config['arch']['loss']['name'] == 'lab_cd':
        return CDLoss()
    elif config['arch']['loss']['name'] == 'ctc_mtl_mono_cd':
        return CTCMtlMonoCDLoss()

    else:
        raise ValueError
