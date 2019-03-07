from nn_.losses.CE import CELoss
from nn_.losses.ctc_phn import CTCPhnLoss


def loss_init(config, model):
    loss_name = config['arch']['loss']['name']
    if loss_name == 'CE':
        return CELoss(model.batch_ordering, config['arch']['loss']['args']['weight_mono'])
        # return MtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    # elif loss_name == 'WaveNet_CE_mtl_mono_cd':
    # return WaveNetMtlMonoCDLoss(config['arch']['loss']['args']['weight_mono'])
    # elif loss_name == 'CE_cd':
    #     return CDLoss()
    # elif loss_name == 'CE_mono':
    #     return MonoLoss()
    elif loss_name == 'CTC':
        return CTCPhnLoss(model.batch_ordering)

    else:
        raise ValueError("Can't find the loss {}".format(loss_name))
