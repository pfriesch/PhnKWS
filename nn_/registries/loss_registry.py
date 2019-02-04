from nn_.losses.ctc_mtl_mono_cd_loss import CTCMtlMonoCDLoss
from nn_.losses.mtl_mono_cd_loss import MtlMonoCDLoss

PADDING_IGNORE_INDEX = -100


def loss_init(config):
    loss_name = config['arch']['loss']['name']
    if loss_name == 'CE_mtl_mono_cd':
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    elif loss_name == 'ctc_mtl_mono_cd':
        return CTCMtlMonoCDLoss()
    else:
        raise ValueError("Can't find the loss {}".format(loss_name))
