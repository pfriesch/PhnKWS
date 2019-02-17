from nn_.networks.CNN_cd_mono import CNN_cd_mono
from nn_.networks.TDNN_cfg_edition_TIMIT import TDNN_cd_mono as TDNN_cd_mono_TIMIT
from nn_.networks.TDNN_cfg_edition_libri import TDNN_cd as TDNN_cd_libri
from nn_.networks.LSTM_cd_mono import LSTM_cd_mono
from nn_.networks.LSTM_phn import LSTM_phn
from nn_.networks.TDNN_mono import TDNN_mono


def model_init(config):
    arch_name = config['arch']['name']
    # if 'decoding' in config and ('normalize_with_counts_from_file' not in config['test']['out_cd']
    #                              or 'lab_cd_num' not in config['arch']['args']):
    if arch_name == "LSTM_cd_mono":
        net = LSTM_cd_mono(input_feat_length=config['arch']['args']['input_feat_length'],
                           input_feat_name=config['arch']['args']['input_feat_name'],
                           lab_cd_num=config['arch']['args']['lab_cd_num'],
                           lab_mono_num=config['arch']['args']['lab_mono_num'])
    elif arch_name == "LSTM_phn":
        net = LSTM_phn(input_feat_length=config['arch']['args']['input_feat_length'],
                       input_feat_name=config['arch']['args']['input_feat_name'],
                       lab_phn_num=config['arch']['args']['lab_phn_num'])

    elif arch_name == "CNN_cd_mono":
        net = CNN_cd_mono(input_feat_length=config['arch']['args']['input_feat_length'],
                          input_feat_name=config['arch']['args']['input_feat_name'],
                          lab_cd_num=config['arch']['args']['lab_cd_num'],
                          lab_mono_num=config['arch']['args']['lab_mono_num'])

    elif arch_name == "TDNN_cd_mono_TIMIT" or arch_name == "TDNN_cd_mono":
        net = TDNN_cd_mono_TIMIT(input_feat_length=config['arch']['args']['input_feat_length'],
                                 input_feat_name=config['arch']['args']['input_feat_name'],
                                 lab_cd_num=config['arch']['args']['lab_cd_num'],
                                 lab_mono_num=config['arch']['args']['lab_mono_num'])
    elif arch_name == "TDNN_cd_libri":
        net = TDNN_cd_libri(input_feat_length=config['arch']['args']['input_feat_length'],
                            input_feat_name=config['arch']['args']['input_feat_name'],
                            lab_cd_num=config['arch']['args']['lab_cd_num'])
    elif arch_name == "TDNN_mono_libri":
        net = TDNN_mono(input_feat_length=config['arch']['args']['input_feat_length'],
                        input_feat_name=config['arch']['args']['input_feat_name'],
                        lab_mono_num=config['arch']['args']['lab_mono_num'])
    else:
        raise ValueError("Can't find the arch {}".format(arch_name))

    return net
