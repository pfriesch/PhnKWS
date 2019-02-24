from nn_.net_modules.WaveNet import WaveNet
from nn_.networks.MLPNet import MLPNet
from nn_.networks.MLP_mtl_Net import MLP_mtl_Net


def model_init(config):
    arch_name = config['arch']['name']
    # if 'decoding' in config and ('normalize_with_counts_from_file' not in config['test']['out_cd']
    #                              or 'lab_cd_num' not in config['arch']['args']):
    # if arch_name == "LSTM_cd_mono":
    #     net = LSTM_cd_mono(input_feat_length=config['arch']['args']['input_feat_length'],
    #                        input_feat_name=config['arch']['args']['input_feat_name'],
    #                        lab_cd_num=config['arch']['args']['lab_cd_num'],
    #                        lab_mono_num=config['arch']['args']['lab_mono_num'])
    # elif arch_name == "LSTM_phn":
    #     net = LSTM_phn(input_feat_length=config['arch']['args']['input_feat_length'],
    #                    input_feat_name=config['arch']['args']['input_feat_name'],
    #                    lab_phn_num=config['arch']['args']['lab_phn_num'])
    #
    # elif arch_name == "CNN_cd_mono":
    #     net = CNN_cd_mono(input_feat_length=config['arch']['args']['input_feat_length'],
    #                       input_feat_name=config['arch']['args']['input_feat_name'],
    #                       lab_cd_num=config['arch']['args']['lab_cd_num'],
    #                       lab_mono_num=config['arch']['args']['lab_mono_num'])

    if arch_name == "MLP_mtl":
        input_feat_name = config['dataset']['features_use'][0]
        input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
            [input_feat_name]['input_feat_length']
        lab_names = config['dataset']['labels_use']
        assert 'lab_cd' in lab_names and 'lab_mono' in lab_names
        # lab_nums = [config['dataset']['dataset_definition']['data_info']['labels'] \
        #                 [lab_name]['num_lab']
        #             for lab_name in lab_names]
        if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
            raise NotImplementedError

        net = MLP_mtl_Net(input_feat_length, input_feat_name,
                          lab_cd_num=config['dataset']['dataset_definition']['data_info']['labels'] \
                                         ['lab_cd']['num_lab'] + 1,
                          lab_mono_num=config['dataset']['dataset_definition']['data_info']['labels'] \
                                           ['lab_mono']['num_lab'] + 1)
    elif arch_name == "MLP":
        input_feat_name = config['dataset']['features_use'][0]
        input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
            [input_feat_name]['input_feat_length']
        lab_num = config['dataset']['dataset_definition']['data_info']['labels'] \
                      [config['dataset']['labels_use'][0]]['num_lab'] + 1
        # if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
        #     lab_num += 1 #TODO
        net = MLPNet(input_feat_length, input_feat_name, lab_num)

    elif arch_name == "WaveNet_mtl":
        input_feat_name = config['dataset']['features_use'][0]
        input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
            [input_feat_name]['input_feat_length']
        lab_names = config['dataset']['labels_use']
        assert 'lab_cd' in lab_names and 'lab_mono' in lab_names
        # lab_nums = [config['dataset']['dataset_definition']['data_info']['labels'] \
        #                 [lab_name]['num_lab']
        #             for lab_name in lab_names]
        if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
            raise NotImplementedError
        net = WaveNet(input_feat_length, input_feat_name, lab_num)

    else:
        raise ValueError("Can't find the arch {}".format(arch_name))

    return net
