from nets.TIMIT_LSTM import TIMIT_LSTM


def model_init(arch_name, fea_index_length, lab_cd_num, use_cuda=False, multi_gpu=False):
    if arch_name == "TIMIT_LSTM":
        net = TIMIT_LSTM(inp_dim=fea_index_length, lab_cd_num=lab_cd_num)

    else:
        raise ValueError

    return net
