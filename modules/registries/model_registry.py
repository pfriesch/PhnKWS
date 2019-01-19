from modules.networks.LSTM_cd_mono import LSTM_cd_mono
from modules.networks.LSTM_phn import LSTM_phn


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
    else:
        raise ValueError("Can't find the arch {}".format(arch_name))

    return net
