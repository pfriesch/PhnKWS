from modules.losses.cd_loss import CDLoss
from modules.losses.ctc_mtl_mono_cd_loss import CTCMtlMonoCDLoss
from modules.losses.ctc_phn import CTCPhnLoss
from modules.losses.mtl_mono_cd_loss import MtlMonoCDLoss


def loss_init(config):
    loss_name = config['arch']['loss']['name']
    if loss_name == 'mtl_mono_cd':
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    elif loss_name == 'lab_cd':
        return CDLoss()
    elif loss_name == 'ctc_mtl_mono_cd':
        return CTCMtlMonoCDLoss()
    elif loss_name == 'ctc_phn':
        return CTCPhnLoss()

    else:
        raise ValueError("Can't find the loss {}".format(loss_name))
