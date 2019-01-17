from data.utils.get_dataset_metadata import get_dataset_metadata
from nets.LSTM_cd_mono import LSTM_cd_mono


def model_init(config):
    arch_name = config['arch']['name']
    if arch_name == "LSTM_cd_mono":
        if 'decoding' in config and ('normalize_with_counts_from_file' not in config['test']['out_cd']
                                     or 'lab_cd_num' not in config['arch']['args']):
            get_dataset_metadata(config)
        net = LSTM_cd_mono(input_feat_length=config['arch']['args']['input_feat_length'],
                           input_feat_name=config['arch']['args']['input_feat_name'],
                           lab_cd_num=config['arch']['args']['lab_cd_num'],
                           lab_mono_num=config['arch']['args']['lab_mono_num'])

    else:
        raise ValueError

    return net
