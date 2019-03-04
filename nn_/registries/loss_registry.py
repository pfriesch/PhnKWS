from nn_.losses.CE_NCL import CELoss_NCL
from nn_.losses.ctc_phn import CTCPhnLoss
from nn_.losses.mtl_mono_cd_loss import MtlMonoCDLoss


def loss_init(config, model):
    loss_name = config['arch']['loss']['name']
    if loss_name == 'CE':
        if model.batch_ordering == "NCL":
            return CELoss_NCL(config['arch']['loss']['args']['weight_mono'])
        else:
            raise NotImplementedError
        return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    # elif loss_name == 'WaveNet_CE_mtl_mono_cd':
    # return WaveNetMtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    # elif loss_name == 'CE_cd':
    #     return CDLoss()
    # elif loss_name == 'CE_mono':
    #     return MonoLoss()
    elif loss_name == 'CTC':
        return CTCPhnLoss()

    else:
        raise ValueError("Can't find the loss {}".format(loss_name))
