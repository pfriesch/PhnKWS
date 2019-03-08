from nn_.networks.MLP import MLP
from nn_.networks.TDNN import TDNN
from nn_.networks.WaveNet import WaveNet


def model_init(config):
    arch_name = config['arch']['name']

    input_feat_name = config['dataset']['features_use'][0]
    input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
        [input_feat_name]['input_feat_length']

    lab_names = config['dataset']['labels_use']
    # assert 'lab_cd' in lab_names and 'lab_mono' in lab_names

    # lab_nums = [config['dataset']['dataset_definition']['data_info']['labels'] \
    #                 [lab_name]['num_lab']
    #             for lab_name in lab_names]
    outputs = {}
    for lab_name in lab_names:
        _out_name = "out_" + lab_name.split("_", 1)[1]
        # Using labels indexed from 1 so 0 is free for padding etc
        outputs[_out_name] = \
            config['dataset']['dataset_definition']['data_info']['labels'][lab_name]['num_lab']

        if config['arch']['loss']['name'] == 'CTC':
            # blank label
            outputs[_out_name] += 1
            # pass
        elif config['arch']['loss']['name'] == 'CE':
            # TODO +1 for padding with 0 etc
            outputs[_out_name] += 1

        else:
            raise NotImplementedError

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
    if arch_name == "MLP":
        # input_feat_name = config['dataset']['features_use'][0]
        # input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
        #     [input_feat_name]['input_feat_length']
        # lab_num = config['dataset']['dataset_definition']['data_info']['labels'] \
        #               [config['dataset']['labels_use'][0]]['num_lab'] + 1
        # # if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
        # #     lab_num += 1 #TODO

        if config['arch']['loss']['name'] == 'CTC':
            # blank label
            _batch_ordering = "NCL"
            if config['training']['dataset_type'] == "SEQUENTIAL_APPENDED_CONTEXT":
                _batch_ordering = "TNCL"
            # pass
        elif config['arch']['loss']['name'] == 'CE':
            # TODO +1 for padding with 0 etc
            _batch_ordering = "NCL"

            if config['training']['dataset_type'] == "FRAMEWISE_SEQUENTIAL_APPENDED_CONTEXT":
                _batch_ordering = "TNCL"

        else:
            raise NotImplementedError

        net = MLP(input_feat_length, input_feat_name, outputs, **config['arch']['args'], batch_ordering=_batch_ordering)


    elif arch_name == "TDNN":

        net = TDNN(input_feat_length, input_feat_name, outputs, **config['arch']['args'])


    # elif arch_name == "WaveNet_mtl_sequential":
    #     assert DatasetType[config['training']['dataset_type']]
    #     dataset_type = DatasetType[config['training']['dataset_type']]
    #     assert dataset_type in [DatasetType.SEQUENTIAL, DatasetType.FRAMEWISE_SEQUENTIAL]
    #
    #     input_feat_name = config['dataset']['features_use'][0]
    #     input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
    #         [input_feat_name]['input_feat_length']
    #     lab_names = config['dataset']['labels_use']
    #     assert 'lab_cd' in lab_names and 'lab_mono' in lab_names
    #     # lab_nums = [config['dataset']['dataset_definition']['data_info']['labels'] \
    #     #                 [lab_name]['num_lab']
    #     #             for lab_name in lab_names]
    #     if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
    #         raise NotImplementedError
    #
    #     net = WaveNet(input_feat_length, input_feat_name,
    #                              lab_cd_num=config['dataset']['dataset_definition']['data_info']['labels'] \
    #                                             ['lab_cd']['num_lab'] + 1,
    #                              lab_mono_num=config['dataset']['dataset_definition']['data_info']['labels'] \
    #                                               ['lab_mono']['num_lab'] + 1)

    elif arch_name == "WaveNet":
        # assert DatasetType[config['training']['dataset_type']]
        # dataset_type = DatasetType[config['training']['dataset_type']]
        # assert dataset_type in [DatasetType.SEQUENTIAL, DatasetType.FRAMEWISE_SEQUENTIAL]

        # input_feat_name = config['dataset']['features_use'][0]
        # input_feat_length = config['dataset']['dataset_definition']['data_info']['features'] \
        #     [input_feat_name]['input_feat_length']
        #
        # lab_names = config['dataset']['labels_use']
        # # assert 'lab_cd' in lab_names and 'lab_mono' in lab_names
        #
        # # lab_nums = [config['dataset']['dataset_definition']['data_info']['labels'] \
        # #                 [lab_name]['num_lab']
        # #             for lab_name in lab_names]
        # outputs = {"out_" + lab_name.split("_", 1)[1]:
        #                config['dataset']['dataset_definition']['data_info']['labels'] \
        #                    [lab_name]['num_lab']
        #            for lab_name in lab_names}

        net = WaveNet(input_feat_length, input_feat_name, outputs, **config['arch']['args'])

    else:
        raise ValueError("Can't find the arch {}".format(arch_name))

    return net
