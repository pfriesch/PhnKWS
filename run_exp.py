##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################
import configparser
import os
import glob
import time

import numpy as np

from utils import configure_logger
from utils.utils import model_init, \
    optimizer_init, lr_scheduler_init, loss_init, set_seed, get_posterior_norm_data, create_chunks, metrics_init
from run_nn import train_on_chunk
from trainer import Trainer
from utils.utils import compute_avg_performance, \
    compute_n_chunks, \
    dump_epoch_results, create_curves, check_environment, read_json, run_shell

check_environment()


def setup_run(config, logger):
    set_seed(config['exp']['seed'])
    #
    # forward_outs = config['forward']['forward_out']
    # forward_normalize_post = config['forward']['normalize_posteriors']
    # forward_count_files = config['forward']['normalize_with_counts_from']
    # require_decodings = config['forward']['require_decoding']

    config, N_out_lab = get_posterior_norm_data(config, logger)
    config["arch"]["args"]["N_out_lab"] = N_out_lab

    # Splitting data into chunks (see out_folder/additional_files)
    create_chunks(config, debug_mode=True)  # TODO debug mode

    # batch_size = None
    # max_seq_length = None
    # if config['exp']['to_do'] == 'train':
    #     max_seq_length = config['batches']['max_seq_length_train']
    #     # *(int(config['exp']['out_info'][-13:-10])+1) # increasing over the epochs
    #     batch_size = config['batches']['batch_size_train']

    # if config['exp']['to_do'] == 'valid':
    #     max_seq_length = config['batches']['max_seq_length_valid']
    #     batch_size = config['batches']['batch_size_valid']
    #
    # if config['exp']['to_do'] == 'forward':
    #     max_seq_length = -1  # do to break forward sentences
    #     batch_size = 1

    # start_time = time.time()

    # data_loader = KaldiDataLoader(config['data_chunk']['fea'],
    #                               config['data_chunk']['lab'],
    #                               config['batches']['batch_size_train'],
    #                               shuffle=False,
    #                               use_gpu=config["exp"]["n_gpu"] > 0,
    #                               num_workers=0)

    # #     # TODO rename save_gpumem to gpu_prefetch maybe?? check first what else save_gpumem does
    # if not (config['exp']['save_gpumem']) and config['exp']['use_cuda']:

    # elapsed_time_reading = time.time() - start_time

    # converting numpy tensors into pytorch tensors and put them on GPUs if specified
    # start_time = time.time()

    # elapsed_time_load = time.time() - start_time

    # Reading model and initialize networks
    # inp_out_dict = config['data_chunk']['fea']

    model = model_init(config['arch']['name'], fea_index_length=config['arch']['args']['fea_index_length'],
                       lab_cd_num=[config['arch']['args']['N_out_lab']['lab_cd']])
    # model = config['model']['model']
    # nns, costs = model_init(inp_out_dict, model, config, config['arch_dict'], config['exp']['use_cuda'],
    #                         config['exp']['multi_gpu'], config['exp']['to_do'])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer_init(config, trainable_params)

    lr_scheduler = lr_scheduler_init(config, optimizer)

    # print(model)
    metrics = metrics_init(config)

    loss = loss_init(config)

    return model, loss, metrics, optimizer, config, lr_scheduler


def main(cfg_file):
    config = read_json(cfg_file)
    set_seed(config['exp']['seed'])

    # Output folder creation
    out_folder = os.path.join(config['exp']['save_dir'], config['exp']['name'])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder + '/exp_files')

    default_logger = configure_logger('default', os.path.join(out_folder, 'log.log'))

    model, loss, metrics, optimizer, config, lr_scheduler = setup_run(config, default_logger)

    trainer = Trainer(model, loss, metrics, optimizer, False, config, True, lr_scheduler, logger=default_logger)
    trainer.train()


if __name__ == '__main__':
    main("cfg/TIMIT_baselines/TIMIT_LSTM_mfcc_refactor.json")
