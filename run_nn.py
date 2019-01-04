##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import random
import time

import numpy as np
import torch
from scipy.ndimage.interpolation import shift

import kaldi_io

from data_loader.kaldi_data_loader import KaldiDataLoader
from trainer import Trainer
from utils.utils import compute_cw_max, model_init, \
    optimizer_init, forward_model, progress, read_json, lr_scheduler_init, loss_init, set_seed




def train_on_chunk(config, cfg_file):
    read_config = read_json(cfg_file)
    assert config == read_config

    model, loss, metrics, optimizer, config, data_loader, lr_scheduler = setup_run(config)

    # pre-training
    # for net in list(nns.keys()):
    #     # TODO Does not actually do pretraining, but provieds an option to load pretrained weights
    #     pt_file_arch = config['arch_dict'][net]['arch_pretrain_file']
    #
    #     if pt_file_arch != 'none':
    #         checkpoint_load = torch.load(pt_file_arch)
    #         nns[net].load_state_dict(checkpoint_load['model_par'])
    #         optimizers[net].load_state_dict(checkpoint_load['optimizer_par'])
    #         # loading lr of the cfg file for pt
    #         optimizers[net].param_groups[0]['lr'] = config['arch_dict'][net]['arch_lr']

    print("test")
    # if config['exp']['to_do'] == 'forward':
    #
    #     post_file = {}
    #     for out_id in range(len(forward_outs)):
    #         if require_decodings[out_id]:
    #             out_file = config['exp']['out_info'].replace('.info', '_' + forward_outs[out_id] + '_to_decode.ark')
    #         else:
    #             out_file = config['exp']['out_info'].replace('.info', '_' + forward_outs[out_id] + '.ark')
    #
    #         post_file[forward_outs[out_id]] = kaldi_io.open_or_fd(out_file, 'wb')


    # ***** Minibatch Processing loop********

    N_snt = len(data_name)
    N_batches = int(N_snt / batch_size)

    beg_batch = 0
    end_batch = batch_size

    snt_index = 0
    beg_snt = 0

    start_time = time.time()

    # array of sentence lengths
    arr_snt_len = shift(shift(data_end_index, -1) - data_end_index, 1)
    arr_snt_len[0] = data_end_index[0]

    loss_sum = 0
    err_sum = 0

    inp_dim = data_set.shape[1]

    for i in range(N_batches):

        max_len = int(max(arr_snt_len[snt_index:snt_index + batch_size]))
        inp = torch.zeros(max_len, batch_size, inp_dim).contiguous()

        for k in range(batch_size):
            snt_len = data_end_index[snt_index] - beg_snt
            N_zeros = max_len - snt_len

            # Appending a random number of initial zeros, tge others are at the end.
            N_zeros_left = random.randint(0, N_zeros)

            # TODO randomizing could have a regularization effect
            inp[N_zeros_left:N_zeros_left + snt_len, k, :] = data_set[beg_snt:beg_snt + snt_len, :]

            beg_snt = data_end_index[snt_index]
            snt_index = snt_index + 1

        # use cuda
        if config['exp']['use_cuda']:
            inp = inp.cuda()

        # Forward input
        outs_dict = forward_model(config['data_chunk']['fea'], config['data_chunk']['lab'], config['arch_dict'],
                                  model, nns, costs, inp, inp_out_dict, max_len,
                                  batch_size,
                                  config['exp']['to_do'], forward_outs)

        if config['exp']['to_do'] == 'train':

            for opt in list(optimizers.keys()):
                optimizers[opt].zero_grad()

            outs_dict['loss_final'].backward()

            # Gradient Clipping (th 0.1) #TODO
            # for net in nns.keys():
            #    torch.nn.utils.clip_grad_norm_(nns[net].parameters(), 0.1)

            for opt in list(optimizers.keys()):
                if not config['arch_dict'][opt]['arch_freeze']:
                    optimizers[opt].step()

        if config['exp']['to_do'] == 'forward':
            for out_id in range(len(forward_outs)):

                # TODO
                out_save = outs_dict[forward_outs[out_id]].data.cpu().numpy()

                if forward_normalize_post[out_id]:
                    # read the config file
                    counts = load_counts(forward_count_files[out_id])
                    out_save = out_save - np.log(counts / np.sum(counts))

                    # save the output
                kaldi_io.write_mat(post_file[forward_outs[out_id]], out_save, data_name[i])
        else:
            loss_sum = loss_sum + outs_dict['loss_final'].detach()
            err_sum = err_sum + outs_dict['err_final'].detach()

        # update it to the next batch
        beg_batch = end_batch
        end_batch = beg_batch + batch_size

        # Progress bar
        if config['exp']['to_do'] == 'train':
            status_string = "Training | (Batch " + str(i + 1) + "/" + str(N_batches) + ")"
        # if config['exp']['to_do'] == 'valid':
        #     status_string = "Validating | (Batch " + str(i + 1) + "/" + str(N_batches) + ")"
        # if config['exp']['to_do'] == 'forward':
        #     status_string = "Forwarding | (Batch " + str(i + 1) + "/" + str(N_batches) + ")"

        progress(i, N_batches, status=status_string)

    elapsed_time_chunk = time.time() - start_time

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    # save the model
    if config['exp']['to_do'] == 'train':

        for net in list(nns.keys()):
            checkpoint = {}
            checkpoint['model_par'] = nns[net].state_dict()
            checkpoint['optimizer_par'] = optimizers[net].state_dict()

            out_file = config['exp']['out_info'].replace('.info', '_' + net + '.pkl')
            torch.save(checkpoint, out_file)

    if config['exp']['to_do'] == 'forward':
        for out_name in forward_outs:
            post_file[out_name].close()

    # Write info file
    with open(config['exp']['out_info'], "w") as text_file:
        text_file.write("[results]\n")
        if config['exp']['to_do'] != 'forward':
            text_file.write("loss=%s\n" % loss_tot.cpu().numpy())
            text_file.write("err=%s\n" % err_tot.cpu().numpy())
        text_file.write("elapsed_time_read=%f (reading dataset)\n" % elapsed_time_reading)
        text_file.write("elapsed_time_load=%f (loading data on pytorch/gpu)\n" % elapsed_time_load)
        text_file.write("elapsed_time_chunk=%f (processing chunk)\n" % elapsed_time_chunk)
        text_file.write("elapsed_time=%f\n" % (elapsed_time_chunk + elapsed_time_load + elapsed_time_reading))
    text_file.close()
