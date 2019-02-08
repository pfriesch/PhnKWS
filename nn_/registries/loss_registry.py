from nn_.losses.cd_loss import CDLoss
from nn_.losses.ctc_mtl_mono_cd_loss import CTCMtlMonoCDLoss
from nn_.losses.ctc_phn import CTCPhnLoss
from nn_.losses.mono_loss import MonoLoss
from nn_.losses.mtl_mono_cd_loss import MtlMonoCDLoss


def loss_init(config):
    loss_name = config['arch']['loss']['name']
    if loss_name == 'CE_mtl_mono_cd':
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    elif loss_name == 'CE_cd':
        return CDLoss()
    elif loss_name == 'CE_mono':
        return MonoLoss()
    elif loss_name == 'ctc_mtl_mono_cd':
        return CTCMtlMonoCDLoss()
    elif loss_name == 'ctc_phn':
        return CTCPhnLoss()

    else:
        raise ValueError("Can't find the loss {}".format(loss_name))
