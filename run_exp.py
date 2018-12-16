##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################
import configparser
import json
import os
import glob
import re

import numpy as np

from run_nn import train_on_chunk
from utils.utils import create_chunks, write_cfg_chunk, compute_avg_performance, \
    compute_n_chunks, \
    dump_epoch_results, create_curves, check_environment, read_json, config2dict, \
    check_and_maybe_replace_output_layer_size_based_on_data, run_shell

check_environment()


def main(cfg_file):
    config = read_json(cfg_file)

    # Output folder creation
    out_folder = config['exp']['out_folder']
    if not os.path.exists(out_folder):
        os.makedirs(out_folder + '/exp_files')

    # Import paths of kaldi libraries
    log_file = config['exp']['out_folder'] + '/log.log'

    # Reading and parsing optional arguments from command line (e.g.,--optimization,lr=0.002)
    # [section_args, field_args, value_args] = read_args_command_line(sys.argv, config)

    # Read, parse, and check the config file
    cfg_file_proto = config['cfg_proto']['cfg_proto']

    # name_arch #<class 'list'>: ['MLP_layers1', 'MLP_layers2', 'MLP_layers3']
    # name_data #<class 'list'>: ['TIMIT_tr', 'TIMIT_dev', 'TIMIT_test']

    # Create the output folder
    if not os.path.exists(out_folder) or not (os.path.exists(out_folder + '/exp_files')):
        os.makedirs(out_folder + '/exp_files')

    config = check_and_maybe_replace_output_layer_size_based_on_data(config)

    name_data = config["datasets"].keys()
    name_arch = config["architectures"].keys()
    print("- Reading config file......OK!")

    # Copy the global cfg file into the output folder

    cfg_file = out_folder + '/conf.json'
    with open(cfg_file, 'w') as configfile:
        json.dump(config, configfile)

    # Splitting data into chunks (see out_folder/additional_files)
    create_chunks(config)

    print("- Chunk creation......OK!\n")

    # create res_file
    res_file_path = out_folder + '/res.res'
    res_file = open(res_file_path, "w")
    res_file.close()

    # Read cfg file options
    cfg_file_proto_chunk = config['cfg_proto']['cfg_proto_chunk']
    # run_nn_script = config['exp']['run_nn_script']
    cmd = config['exp']['cmd']
    N_ep = int(config['exp']['n_epochs_tr'])
    tr_data_lst = config['data_use']['train_with']
    valid_data_lst = config['data_use']['valid_with']
    forward_data_lst = config['data_use']['forward_with']
    max_seq_length_train = int(config['batches']['max_seq_length_train'])
    forward_save_files = config['forward']['save_out_file']
    assert isinstance(forward_save_files, bool)

    # Learning rates and architecture-specific optimization parameters
    lr = {}
    improvement_threshold = {}
    halving_factor = {}
    pt_files = {}

    for arch in config["architectures"]:
        lr[arch] = config["architectures"][arch]['arch_lr']
        improvement_threshold[arch] = config["architectures"][arch]['arch_improvement_threshold']
        halving_factor[arch] = config["architectures"][arch]['arch_halving_factor']
        pt_files[arch] = config["architectures"][arch]['arch_pretrain_file']

    if config['batches']['increase_seq_length_train']:
        max_seq_length_train_curr = int(config['batches']['start_seq_len_train'])
    else:
        max_seq_length_train_curr = max_seq_length_train

    # --------TRAINING LOOP--------#
    for ep in range(N_ep):

        tr_loss_tot = 0
        tr_error_tot = 0
        tr_time_tot = 0

        print("----------------------------- Epoch {:03d} / {:03d} -----------------------------".format(ep, N_ep - 1))

        for tr_data in tr_data_lst:
            # #### PREPERATION ####
            # 1. info file
            # loss=6.0169883
            # err=0.83695066
            # elapsed_time_read=13.312386 (reading dataset)
            # elapsed_time_load=11.523646 (loading data on pytorch/gpu)
            # elapsed_time_chunk=51.931287 (processing chunk)
            # elapsed_time=76.767319

            # 2 data list scp per chunk

            # model backup after each chunk

            # 3 save config to run only that chunk

            # run training

            # save chunk model state

            # remove previous state

            # compute avg train acc and loss

            # Compute the total number of chunks for each training epoch
            N_ck_tr = compute_n_chunks(out_folder, tr_data, ep, 'train')

            # ***Epoch training***
            for ck in range(N_ck_tr):

                # path of the list of features for this chunk
                lst_file = out_folder + '/exp_files/train_' + tr_data + '_ep' + format(ep, "03d") + '_ck' + format(ck,
                                                                                                                   "02d") + '_*.lst'

                # paths of the output files (info,model,chunk_specific cfg file)
                info_file_name = out_folder + '/exp_files/train_' + tr_data + '_ep' + format(ep,
                                                                                             "03d") + '_ck' + format(ck,
                                                                                                                     "02d") + '.info'

                if ep + ck == 0:
                    model_files_past = {}
                else:
                    model_files_past = model_files

                model_files = {}
                for arch in list(pt_files.keys()):
                    model_files[arch] = info_file_name.replace('.info', '_' + arch + '.pkl')

                config_chunk_file = "{}/exp_files/train_{}_ep{:03d}_ck{:02d}.json" \
                    .format(out_folder, tr_data, ep, ck)

                # TODO run the experiement with json config, but keep ini config around to make sure it is equal or to helt wtf is going on
                # TODO run with custom arch, optimizer, lrscheduler, loss

                # Write chunk-specific cfg file
                write_cfg_chunk(cfg_file, config_chunk_file,
                                pt_files, lst_file,
                                info_file_name,
                                'train',
                                tr_data, lr, max_seq_length_train_curr, name_data, ep, ck)

                # if this chunk has not already been processed, do training...
                if not (os.path.exists(info_file_name)):
                    print('Training %s chunk = %i / %i' % (tr_data, ck + 1, N_ck_tr))

                    # Doing training
                    train_on_chunk(config_chunk_file)

                # update pt_file (used to initialized the DNN for the next chunk)
                for pt_arch in list(pt_files.keys()):
                    pt_files[pt_arch] = "{}/exp_files/train_{}_ep{:03d}_ck{:02d}_{}.pkl" \
                        .format(out_folder, tr_data, int(ep), int(ck), pt_arch)

                # remove previous pkl files
                if len(list(model_files_past.keys())) > 0:
                    for pt_arch in list(pt_files.keys()):
                        if os.path.exists(model_files_past[pt_arch]):
                            os.remove(model_files_past[pt_arch])

            # Training Loss and Error
            tr_info_lst = sorted(
                glob.glob(out_folder + '/exp_files/train_' + tr_data + '_ep' + format(ep, "03d") + '*.info'))
            [tr_loss, tr_error, tr_time] = compute_avg_performance(tr_info_lst)

            tr_loss_tot = tr_loss_tot + tr_loss
            tr_error_tot = tr_error_tot + tr_error
            tr_time_tot = tr_time_tot + tr_time

            # ***Epoch validation***
            if ep > 0:
                # store previous-epoch results (useful for learnig rate anealling)
                valid_peformance_dict_prev = valid_peformance_dict

            valid_peformance_dict = {}
            tot_time = tr_time

        for valid_data in valid_data_lst:

            # Compute the number of chunks for each validation dataset
            N_ck_valid = compute_n_chunks(out_folder, valid_data, ep, 'valid')

            for ck in range(N_ck_valid):

                # path of the list of features for this chunk
                lst_file = out_folder + '/exp_files/valid_' + valid_data + '_ep' + format(ep, "03d") + '_ck' + format(
                    ck,
                    "02d") + '_*.lst'

                # paths of the output files
                info_file_name = out_folder + '/exp_files/valid_' + valid_data + '_ep' + format(ep,
                                                                                                "03d") + '_ck' + format(
                    ck,
                    "02d") + '.info'
                config_chunk_file_name = out_folder + '/exp_files/valid_' + valid_data + '_ep' + format(ep,
                                                                                                        "03d") + '_ck' + format(
                    ck, "02d") + '.json'

                # Write chunk-specific cfg file
                write_cfg_chunk(cfg_file, config_chunk_file_name,
                                model_files, lst_file,
                                info_file_name,
                                'valid',
                                valid_data, lr, max_seq_length_train_curr, name_data, ep, ck)

                # Do validation if the chunk was not already processed
                if not (os.path.exists(info_file_name)):
                    print('Validating %s chunk = %i / %i' % (valid_data, ck + 1, N_ck_valid))

                    # Doing eval
                    train_on_chunk(config_chunk_file_name)

            # Compute validation performance
            valid_info_lst = sorted(
                glob.glob(out_folder + '/exp_files/valid_' + valid_data + '_ep' + format(ep, "03d") + '*.info'))
            [valid_loss, valid_error, valid_time] = compute_avg_performance(valid_info_lst)
            valid_peformance_dict[valid_data] = [valid_loss, valid_error, valid_time]
            tot_time = tot_time + valid_time

        # Print results in both res_file and stdout
        dump_epoch_results(res_file_path, ep, tr_data_lst, tr_loss_tot, tr_error_tot, tot_time, valid_data_lst,
                           valid_peformance_dict, lr, N_ep)

        #  if needed, update sentence_length
        if config['batches']['increase_seq_length_train']:
            max_seq_length_train_curr = max_seq_length_train_curr * int(
                config['batches']['multply_factor_seq_len_train'])
            if max_seq_length_train_curr > max_seq_length_train:
                max_seq_length_train_curr = max_seq_length_train

        # Check for learning rate annealing
        if ep > 0:
            # computing average validation error (on all the dataset specified)
            err_valid_mean = np.mean(np.asarray(list(valid_peformance_dict.values()))[:, 1])
            err_valid_mean_prev = np.mean(np.asarray(list(valid_peformance_dict_prev.values()))[:, 1])

            for lr_arch in list(lr.keys()):
                if ((err_valid_mean_prev - err_valid_mean) / err_valid_mean) < improvement_threshold[lr_arch]:
                    lr[lr_arch] = lr[lr_arch] * halving_factor[lr_arch]

    # --------FORWARD--------#
    for forward_data in forward_data_lst:

        # Compute the number of chunks
        N_ck_forward = compute_n_chunks(out_folder, forward_data, ep, 'forward')

        for ck in range(N_ck_forward):

            print('Testing %s chunk = %i / %i' % (forward_data, ck + 1, N_ck_forward))

            # path of the list of features for this chunk
            lst_file = out_folder + '/exp_files/forward_' + forward_data + '_ep' + format(ep, "03d") + '_ck' + format(
                ck,
                "02d") + '_*.lst'

            # output file
            info_file_name = out_folder + '/exp_files/forward_' + forward_data + '_ep' + format(ep,
                                                                                                "03d") + '_ck' + format(
                ck,
                "02d") + '.info'
            config_chunk_file_name = out_folder + '/exp_files/forward_' + forward_data + '_ep' + format(ep,
                                                                                                        "03d") + '_ck' + format(
                ck, "02d") + '.json'

            # Write chunk-specific cfg file
            write_cfg_chunk(cfg_file, config_chunk_file_name,
                            model_files, lst_file,
                            info_file_name,
                            'forward',
                            forward_data, lr, max_seq_length_train_curr, name_data, ep, ck)

            # Do forward if the chunk was not already processed
            if not (os.path.exists(info_file_name)):
                # Doing forward
                train_on_chunk(config_chunk_file_name)

    # --------DECODING--------#
    dec_lst = glob.glob(out_folder + '/exp_files/*_to_decode.ark')
    assert len(dec_lst) > 0

    forward_data_lst = config['data_use']['forward_with']
    forward_outs = config['forward']['forward_out']
    forward_dec_outs = config['forward']['require_decoding']

    for data in forward_data_lst:
        for k in range(len(forward_outs)):
            if forward_dec_outs[k]:

                print('Decoding %s output %s' % (data, forward_outs[k]))

                info_file_name = out_folder + '/exp_files/decoding_' + data + '_' + forward_outs[k] + '.info'

                # create decode config file
                config_dec_file = out_folder + '/decoding_' + data + '_' + forward_outs[k] + '.conf'
                config_dec = configparser.ConfigParser()
                config_dec.add_section('decoding')

                for dec_key in list(config['decoding'].keys()):
                    config_dec.set('decoding', dec_key, str(config['decoding'][dec_key]))

                lab_to_use = 'lab_cd'  # TODO just choosing one here since they should be the same, smells!
                config_dec.set('decoding', 'alidir',
                               os.path.abspath(config['datasets'][data]['lab'][lab_to_use]['lab_folder']))
                config_dec.set('decoding', 'data',
                               os.path.abspath(config['datasets'][data]['lab'][lab_to_use]['lab_data_folder']))
                config_dec.set('decoding', 'graphdir',
                               os.path.abspath(config['datasets'][data]['lab'][lab_to_use]['lab_graph']))

                with open(config_dec_file, 'w') as configfile:
                    config_dec.write(configfile)

                out_folder = os.path.abspath(out_folder)
                files_dec = out_folder + '/exp_files/forward_' + data + '_ep*_ck*_' + forward_outs[k] + '_to_decode.ark'
                out_dec_folder = out_folder + '/decode_' + data + '_' + forward_outs[k]

                if not (os.path.exists(info_file_name)):

                    # Run the decoder
                    cmd_decode = cmd + config['decoding']['decoding_script_folder'] + '/' + config['decoding'][
                        'decoding_script'] + ' ' + os.path.abspath(
                        config_dec_file) + ' ' + out_dec_folder + ' \"' + files_dec + '\"'
                    print("cwd:", os.getcwd())
                    run_shell(cmd_decode, log_file)

                    # remove ark files if needed
                    if not forward_save_files:
                        list_rem = glob.glob(files_dec)
                        for rem_ark in list_rem:
                            os.remove(rem_ark)

                # Print WER results and write info file
                cmd_res = './scripts/check_res_dec.sh ' + out_dec_folder
                wers = run_shell(cmd_res, log_file).decode('utf-8')
                res_file = open(res_file_path, "a")
                res_file.write('%s\n' % wers)
                print(wers)

    # Saving Loss and Err as .txt and plotting curves
    create_curves(out_folder, N_ep, valid_data_lst)


if __name__ == '__main__':
    main("cfg/TIMIT_baselines/TIMIT_MLP_mfcc.json")
