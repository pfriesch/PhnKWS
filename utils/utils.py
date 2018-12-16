##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import configparser
import sys
import os.path
import random
import subprocess
import re
import glob
from distutils.util import strtobool
import importlib
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from jsmin import jsmin

from neural_networks.modules.MLP import MLP
from neural_networks.modules.CNN import CNN


def config2dict(config):
    return {s: dict(config.items(s)) for s in config.sections()}


def read_json(path):
    if not (os.path.exists(path)):
        raise ValueError('ERROR: The json file {} does not exist!\n'.format(path))
    else:
        with open(path, "r") as js_file:
            _dict = json.loads(jsmin(js_file.read()))
    return _dict


def write_json(_dict, path):
    assert isinstance(_dict, dict)
    # TODO check for overwrite, maybe?
    with open(path, "w") as js_file:
        json.dump(_dict, js_file, indent=1)


def which(program):
    """https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    """
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def check_environment():
    assert os.environ['KALDI_ROOT']

    PATH = os.environ['PATH']

    assert "tools/openfst" in PATH and "src/featbin" in PATH and "src/gmmbin" in PATH and "src/bin" in PATH and "src/nnetbin" in PATH

    assert isinstance(which("hmm-info"), str), which("hmm-info")

    # """
    # export KALDI_ROOT=`pwd`/../../..
    # export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
    # [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
    # . $KALDI_ROOT/tools/config/common_path.sh
    # export LC_ALL=C
    #
    # """


# def run_command(cmd):
#     """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
#     """
#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     stdout = []
#     while True:
#         line = p.stdout.readline()
#         stdout.append(line)
#         print((line.decode("utf-8")))
#         if line == '' and p.poll() != None:
#             break
#     return ''.join(stdout)


# def run_shell_display(cmd):
#     p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#     while True:
#         out = p.stdout.read(1).decode('utf-8')
#         if out == '' and p.poll() != None:
#             break
#         if out != '':
#             sys.stdout.write(out)
#             sys.stdout.flush()
#     return


def run_shell(cmd, log_file):
    print("RUN: ", cmd)
    if cmd.split(" ")[0].endswith(".sh"):
        if not (os.path.isfile(cmd.split(" ")[0]) and os.access(cmd.split(" ")[0], os.X_OK)):
            print("WARNING: {} does not exist or is not runnable!".format(cmd.split(" ")[0]))

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()
    p.wait()
    print("OUTPUT: ", output.decode("utf-8"))
    print("ERROR:  ", err.decode("utf-8"), file=sys.stderr)
    with open(log_file, 'a+') as logfile:
        logfile.write(output.decode("utf-8") + '\n')
        logfile.write(err.decode("utf-8") + '\n')
    return output


# def read_args_command_line(args, config):
#     sections = []
#     fields = []
#     values = []
#
#     for i in range(2, len(args)):
#         # check if the option is valid
#         r = re.compile('--.*,.*=.*')
#         if r.match(args[i]) is None:
#             raise ValueError(
#                 'ERROR: option \"%s\" from command line is not valid! (the format must be \"--section,field=value\")\n' % (
#                     args[i]))
#
#         sections.append(re.search('--(.*),', args[i]).group(1))
#         fields.append(re.search(',(.*)=', args[i]).group(1))
#         values.append(re.search('=(.*)', args[i]).group(1))
#
#     # parsing command line arguments
#     for i in range(len(sections)):
#         if sections[i] in config.sections():
#             if fields[i] in list(config[sections[i]]):
#                 config[sections[i]][fields[i]] = values[i]
#             else:
#                 raise ValueError('ERROR: field \"%s\" of section \"%s\" from command line is not valid!")\n' % (
#                     fields[i], sections[i]))
#         else:
#             raise ValueError('ERROR: section \"%s\" from command line is not valid!")\n' % (sections[i]))
#
#     return [sections, fields, values]


def compute_avg_performance(info_lst):
    losses = []
    errors = []
    times = []

    for tr_info_file in info_lst:
        config_res = configparser.ConfigParser()
        config_res.read(tr_info_file)
        losses.append(float(config_res['results']['loss']))
        errors.append(float(config_res['results']['err']))
        times.append(float(config_res['results']['elapsed_time']))

    loss = np.mean(losses)
    error = np.mean(errors)
    time = np.sum(times)

    return [loss, error, time]


def check_and_maybe_replace_output_layer_size_based_on_data(config):
    out_folder = config['exp']['out_folder']

    # Parsing forward field
    model = config['model']['model']
    possible_outs = [layer['out_name'] for layer in model]
    forward_out_lst = config['forward']['forward_out']
    forward_norm_lst = config['forward']['normalize_with_counts_from']
    forward_norm_bool_lst = config['forward']['normalize_posteriors']

    train_dataset_lab = config['datasets'][config['data_use']['train_with'][0]]['lab']
    lab_lst = list(train_dataset_lab.keys())
    N_out_lab = {k: 'none' for k in lab_lst}

    for i in range(len(forward_out_lst)):
        if forward_out_lst[i] not in possible_outs:
            raise ValueError(
                """ERROR: the output "{}" in the section "forwad_out" is not defined in section model)\n"""
                    .format(forward_out_lst[i]))

        if forward_norm_bool_lst[i]:

            if forward_norm_lst[i] not in lab_lst:
                if not os.path.exists(forward_norm_lst[i]):
                    raise ValueError(
                        """ERROR: the count_file "{}" in the section "forwad_out" is does not exist)\n"""
                            .format(forward_norm_lst[i]))

                else:
                    # Check if the specified file is in the right format
                    f = open(forward_norm_lst[i], "r")
                    cnts = f.read()
                    if not (bool(re.match("(.*)\[(.*)\]", cnts))):
                        raise ValueError(
                            """ERROR: the count_file "{}" in the section "forwad_out" is not in the right format)\n"""
                                .format(forward_norm_lst[i]))


            else:
                # Try to automatically retrieve the config file
                if "ali-to-pdf" in train_dataset_lab[forward_norm_lst[i]]['lab_opts']:
                    log_file = config['exp']['out_folder'] + '/log.log'
                    folder_lab_count = train_dataset_lab[forward_norm_lst[i]]['lab_folder']
                    cmd = "hmm-info " + folder_lab_count + "/final.mdl | awk '/pdfs/{print $4}'"
                    output = run_shell(cmd, log_file)
                    N_out = int(output.decode().rstrip())
                    N_out_lab[forward_norm_lst[i]] = N_out
                    count_file_path = out_folder + '/exp_files/forward_' + forward_out_lst[i] + '_' + forward_norm_lst[
                        i] + '.count'
                    cmd = "analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim=" + str(
                        N_out) + " \"ark:ali-to-pdf " + folder_lab_count + "/final.mdl \\\"ark:gunzip -c " + folder_lab_count + "/ali.*.gz |\\\" ark:- |\" " + count_file_path
                    run_shell(cmd, log_file)
                    forward_norm_lst[i] = count_file_path

                else:
                    raise ValueError(
                        """ERROR: Not able to automatically retrieve count file for the label""" +
                        """ "{}". Please add a valid count file path in "normalize_with_counts_from" or set normalize_posteriors=False \n"""
                        .format(forward_norm_lst[i]))

    # Update the config file with the count_file paths
    config['forward']['normalize_with_counts_from'] = forward_norm_lst

    # When possible replace the pattern "N_out_lab*" with the detected number of output

    for arch in config['architectures']:
        for opt_key, opt_val in config['architectures'][arch].items():
            for _lst in lab_lst:
                pattern = 'N_out_' + _lst
                if isinstance(opt_val, str) and pattern in opt_val:
                    if N_out_lab[_lst] != 'none':
                        # TODO handle multiple inputs
                        config['architectures'][arch][opt_key] = [N_out_lab[_lst]]
                    else:
                        raise ValueError(
                            'ERROR: Cannot automatically retrieve the number of output in %s. Plese, add manually the number of outputs \n' % (
                                pattern))

    return config


# TODO check config

def split_chunks(seq, size):
    newseq = []
    splitsize = 1.0 / size * len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
    return newseq


def create_chunks(json_config):
    # splitting data into chunks (see out_folder/additional_files)
    out_folder = json_config['exp']['out_folder']
    seed = int(json_config['exp']['seed'])
    N_ep = int(json_config['exp']['n_epochs_tr'])

    # Setting the random seed
    random.seed(seed)

    # training chunk lists creation
    tr_data_name = json_config['data_use']['train_with']

    # Reading validation feature lists
    for dataset in tr_data_name:
        fea_names = list(json_config['datasets'][dataset]['fea'].keys())
        list_fea = [json_config['datasets'][dataset]['fea'][feats]['fea_lst'] for feats in
                    json_config['datasets'][dataset]['fea']]

        # assert [fea_names, list_fea, fea_opts, cws_left, cws_right] == parse_fea_field(
        #     config[cfg_item2sec(config, 'data_name', dataset)]['fea'])

        N_chunks = int(json_config['datasets'][dataset]['n_chunks'])

        full_list = []

        for i in range(len(fea_names)):
            full_list.append([line.rstrip('\n') + ',' for line in open(list_fea[i])])
            full_list[i] = sorted(full_list[i])

        # concatenating all the featues in a single file (useful for shuffling consistently)
        full_list_fea_conc = full_list[0]
        for i in range(1, len(full_list)):
            full_list_fea_conc = list(map(str.__add__, full_list_fea_conc, full_list[i]))

        for ep in range(N_ep):
            #  randomize the list
            random.shuffle(full_list_fea_conc)
            tr_chunks_fea = list(split_chunks(full_list_fea_conc, N_chunks))
            tr_chunks_fea.reverse()

            for ck in range(N_chunks):
                for i in range(len(fea_names)):

                    tr_chunks_fea_split = []
                    for snt in tr_chunks_fea[ck]:
                        # print(snt.split(',')[i])
                        tr_chunks_fea_split.append(snt.split(',')[i])

                    output_lst_file = out_folder + '/exp_files/train_' + dataset + '_ep' + format(ep,
                                                                                                  "03d") + '_ck' + format(
                        ck, "02d") + '_' + fea_names[i] + '.lst'
                    f = open(output_lst_file, 'w')
                    tr_chunks_fea_wr = [x + '\n' for x in tr_chunks_fea_split]
                    f.writelines(tr_chunks_fea_wr)
                    f.close()

    # Validation chunk lists creation
    valid_data_name = json_config['data_use']['valid_with']

    # Reading validation feature lists
    for dataset in valid_data_name:
        fea_names = list(json_config['datasets'][dataset]['fea'].keys())
        list_fea = [json_config['datasets'][dataset]['fea'][feats]['fea_lst'] for feats in
                    json_config['datasets'][dataset]['fea']]

        N_chunks = int(json_config['datasets'][dataset]['n_chunks'])

        full_list = []

        for i in range(len(fea_names)):
            full_list.append([line.rstrip('\n') + ',' for line in open(list_fea[i])])
            full_list[i] = sorted(full_list[i])

        # concatenating all the featues in a single file (useful for shuffling consistently)
        full_list_fea_conc = full_list[0]
        for i in range(1, len(full_list)):
            full_list_fea_conc = list(map(str.__add__, full_list_fea_conc, full_list[i]))

        # randomize the list
        random.shuffle(full_list_fea_conc)
        valid_chunks_fea = list(split_chunks(full_list_fea_conc, N_chunks))

        for ep in range(N_ep):
            for ck in range(N_chunks):
                for i in range(len(fea_names)):

                    valid_chunks_fea_split = [];
                    for snt in valid_chunks_fea[ck]:
                        # print(snt.split(',')[i])
                        valid_chunks_fea_split.append(snt.split(',')[i])

                    output_lst_file = out_folder + '/exp_files/valid_' + dataset + '_ep' + format(ep,
                                                                                                  "03d") + '_ck' + format(
                        ck, "02d") + '_' + fea_names[i] + '.lst'
                    f = open(output_lst_file, 'w')
                    valid_chunks_fea_wr = [x + '\n' for x in valid_chunks_fea_split]
                    f.writelines(valid_chunks_fea_wr)
                    f.close()

    # forward chunk lists creation
    forward_data_name = json_config['data_use']['forward_with']

    # Reading validation feature lists
    for dataset in forward_data_name:
        fea_names = list(json_config['datasets'][dataset]['fea'].keys())
        list_fea = [json_config['datasets'][dataset]['fea'][feats]['fea_lst'] for feats in
                    json_config['datasets'][dataset]['fea']]

        N_chunks = int(json_config['datasets'][dataset]['n_chunks'])

        full_list = []

        for i in range(len(fea_names)):
            full_list.append([line.rstrip('\n') + ',' for line in open(list_fea[i])])
            full_list[i] = sorted(full_list[i])

        # concatenating all the featues in a single file (useful for shuffling consistently)
        full_list_fea_conc = full_list[0]
        for i in range(1, len(full_list)):
            full_list_fea_conc = list(map(str.__add__, full_list_fea_conc, full_list[i]))

        # randomize the list
        random.shuffle(full_list_fea_conc)
        forward_chunks_fea = list(split_chunks(full_list_fea_conc, N_chunks))

        for ck in range(N_chunks):
            for i in range(len(fea_names)):

                forward_chunks_fea_split = []
                for snt in forward_chunks_fea[ck]:
                    # print(snt.split(',')[i])
                    forward_chunks_fea_split.append(snt.split(',')[i])

                output_lst_file = out_folder + '/exp_files/forward_' + dataset + '_ep' + format(ep,
                                                                                                "03d") + '_ck' + format(
                    ck, "02d") + '_' + fea_names[i] + '.lst'
                f = open(output_lst_file, 'w')
                forward_chunks_fea_wr = [x + '\n' for x in forward_chunks_fea_split]
                f.writelines(forward_chunks_fea_wr)
                f.close()


def write_cfg_chunk(json_cfg_file, json_config_chunk_file, pt_file,
                    lst_file, info_file, to_do,
                    data_set_name, lr, max_seq_length_train_curr, name_data, ep, ck):
    # writing the chunk-specific cfg file
    config = read_json(json_cfg_file)
    config_chunk = read_json(json_cfg_file)

    # Exp section
    config_chunk['exp']['to_do'] = to_do
    config_chunk['exp']['out_info'] = info_file

    # change seed for randomness
    config_chunk['exp']['seed'] = str(int(config_chunk['exp']['seed']) + ep + ck)

    for arch in pt_file:
        config_chunk["architectures"][arch]['arch_pretrain_file'] = pt_file[arch]

    # writing the current learning rate
    for arch in lr:
        config_chunk["architectures"][arch]['arch_lr'] = lr[arch]

    # Data_chunk section
    config_chunk['data_chunk'] = config['datasets'][data_set_name]

    # TODO make feat type independend, still shake debug with more than one feature list
    lst_files = sorted(glob.glob(lst_file))
    assert len(lst_files) == 1
    mfcc_lst_file = lst_files[0]
    for lst_file in lst_files:
        assert "mfcc" in lst_file

    config_chunk['data_chunk']['fea']["mfcc"]['fea_lst'] = mfcc_lst_file
    del config_chunk['data_chunk']['n_chunks']
    del config_chunk['decoding']
    del config_chunk['data_use']
    del config_chunk['batches']['increase_seq_length_train']
    del config_chunk['batches']['start_seq_len_train']
    del config_chunk['batches']['multply_factor_seq_len_train']

    config_chunk['batches']['max_seq_length_train'] = max_seq_length_train_curr

    write_json(config_chunk, json_config_chunk_file)


def compute_n_chunks(out_folder, data_list, ep, step):
    list_ck = sorted(
        glob.glob(out_folder + '/exp_files/' + step + '_' + data_list + '_ep' + format(ep, "03d") + '*.lst'))
    last_ck = list_ck[-1]
    N_ck = int(re.findall('_ck(.+)_', last_ck)[-1].split('_')[0]) + 1
    return N_ck


def parse_model_field(cfg_file):
    # Reading the config file
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # reading the proto file
    model_proto_file = config['model']['model_proto']
    f = open(model_proto_file, "r")
    proto_model = f.read()

    # readiing the model string
    model = config['model']['model']

    # Reading fea,lab arch architectures from the cfg file   
    fea_lst = list(re.findall('fea_name=(.*)\n', config['dataset1']['fea'].replace(' ', '')))
    lab_lst = list(re.findall('lab_name=(.*)\n', config['dataset1']['lab'].replace(' ', '')))
    arch_lst = list(re.findall('arch_name=(.*)\n', open(cfg_file, 'r').read().replace(' ', '')))
    possible_operations = re.findall('(.*)\((.*),(.*)\)\n', proto_model)

    possible_inputs = fea_lst
    model_arch = list([_f for _f in model.replace(' ', '').split('\n') if _f])

    # Reading the model field line by line
    for line in model_arch:

        pattern = '(.*)=(.*)\((.*),(.*)\)'

        if not re.match(pattern, line):
            raise ValueError(
                'ERROR: all the entries must be of the following type: output=operation(str,str), got (%s)\n' % (line))

        else:

            # Analyze line and chech if it is compliant with proto_model
            [layer['out_name'], operation, layer['inp1'], layer['inp2']] = list(re.findall(pattern, line)[0])
            inps = [layer['inp1'], layer['inp2']]

            found = False
            for i in range(len(possible_operations)):
                if operation == possible_operations[i][0]:
                    found = True

                    for k in range(1, 3):
                        if possible_operations[i][k] == 'architecture':
                            if inps[k - 1] not in arch_lst:
                                raise ValueError(
                                    'ERROR: the architecture \"%s\" is not in the architecture lists of the config file (possible architectures are %s)\n' % (
                                        inps[k - 1], arch_lst))

                        if possible_operations[i][k] == 'label':
                            if inps[k - 1] not in lab_lst:
                                raise ValueError(
                                    'ERROR: the label \"%s\" is not in the label lists of the config file (possible labels are %s)\n' % (
                                        inps[k - 1], lab_lst))

                        if possible_operations[i][k] == 'input':
                            if inps[k - 1] not in possible_inputs:
                                raise ValueError(
                                    'ERROR: the input \"%s\" is not defined before (possible inputs are %s)\n' % (
                                        inps[k - 1], possible_inputs))

                        if possible_operations[i][k] == 'float':

                            try:
                                float(inps[k - 1])
                            except ValueError:
                                raise ValueError(
                                    'ERROR: the input \"%s\" must be a float, got %s\n' % (inps[k - 1], line))

                                # Update the list of possible inpus
                    possible_inputs.append(layer['out_name'])
                    break

            if found == False:
                raise ValueError(
                    ('ERROR: operation \"%s\" does not exists (not defined into the model proto file)\n' % (operation)))

    # Check for the mandatory fiels
    if 'loss_final' not in "".join(model_arch):
        raise ValueError('ERROR: the variable loss_final should be defined in model\n')

    if 'err_final' not in "".join(model_arch):
        raise ValueError('ERROR: the variable err_final should be defined in model\n')


def terminal_node_detection(model_arch, node):
    terminal = True
    pattern = '(.*)=(.*)\((.*),(.*)\)'

    for line in model_arch:
        [layer['out_name'], operation, layer['inp1'], layer['inp2']] = list(re.findall(pattern, line)[0])
        if layer['inp1'] == node or layer['inp2'] == node:
            terminal = False

    return terminal


def create_block_connection(lst_inp, model_arch, diag_lines, cnt_names, arch_dict):
    if lst_inp == []:
        return [[], [], diag_lines]

    pattern = '(.*)=(.*)\((.*),(.*)\)'

    arch_current = []
    output_conn = []
    current_inp = []

    for input_element in lst_inp:

        for l in range(len(model_arch)):
            [layer['out_name'], operation, layer['inp1'], layer['inp2']] = list(re.findall(pattern, model_arch[l])[0])

            if layer['inp1'] == input_element or layer['inp2'] == input_element:
                if operation == 'compute':
                    arch_current.append(layer['inp1'])
                    output_conn.append(layer['out_name'])
                    current_inp.append(layer['inp2'])
                    model_arch[l] = 'processed' + '=' + operation + '(' + layer['inp1'] + ',processed)'
                else:
                    arch_current.append(layer['out_name'])
                    output_conn.append(layer['out_name'])
                    if layer['inp1'] == input_element:
                        current_inp.append(layer['inp1'])
                        model_arch[l] = layer['out_name'] + '=' + operation + '(processed,' + layer['inp2'] + ')'

                    if layer['inp2'] == input_element:
                        current_inp.append(layer['inp2'])
                        model_arch[l] = layer['out_name'] + '=' + operation + '(' + layer['inp1'] + ',processed)'

    for i in range(len(arch_current)):
        # Create connections
        diag_lines = diag_lines + str(cnt_names.index(arch_dict[current_inp[i]])) + ' -> ' + str(
            cnt_names.index(arch_current[i])) + ' [label = "' + current_inp[i] + '"]\n'

    #  remove terminal nodes from output list
    output_conn_pruned = []
    for node in output_conn:
        if not (terminal_node_detection(model_arch, node)):
            output_conn_pruned.append(node)

    [arch_current, output_conn, diag_lines] = create_block_connection(output_conn, model_arch, diag_lines, cnt_names,
                                                                      arch_dict)

    return [arch_current, output_conn_pruned, diag_lines]


def create_block_diagram(cfg_file):
    # Reading the config file
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # readiing the model string
    model = config['model']['model']

    # Reading fea,lab arch architectures from the cfg file   
    pattern = '(.*)=(.*)\((.*),(.*)\)'

    fea_lst = list(re.findall('fea_name=(.*)\n', config['dataset1']['fea'].replace(' ', '')))
    lab_lst = list(re.findall('lab_name=(.*)\n', config['dataset1']['lab'].replace(' ', '')))
    arch_lst = list(re.findall('arch_name=(.*)\n', open(cfg_file, 'r').read().replace(' ', '')))

    out_diag_file = config['exp']['out_folder'] + '/model.diag'

    model_arch = list([_f for _f in model.replace(' ', '').split('\n') if _f])

    diag_lines = 'blockdiag {\n';

    cnt = 0
    cnt_names = []
    arch_lst = []
    fea_lst_used = []
    lab_lst_used = []

    for line in model_arch:
        if 'err_final=' in line:
            model_arch.remove(line)

    # Initializations of the blocks
    for line in model_arch:
        [layer['out_name'], operation, layer['inp1'], layer['inp2']] = list(re.findall(pattern, line)[0])

        if operation != 'compute':

            # node architecture
            diag_lines = diag_lines + str(cnt) + ' [label="' + operation + '",shape = roundedbox];\n'
            arch_lst.append(layer['out_name'])
            cnt_names.append(layer['out_name'])
            cnt = cnt + 1

            # labels
            if layer['inp2'] in lab_lst:
                diag_lines = diag_lines + str(cnt) + ' [label="' + layer['inp2'] + '",shape = roundedbox];\n'
                if layer['inp2'] not in lab_lst_used:
                    lab_lst_used.append(layer['inp2'])
                    cnt_names.append(layer['inp2'])
                    cnt = cnt + 1

            # features
            if layer['inp1'] in fea_lst:
                diag_lines = diag_lines + str(cnt) + ' [label="' + layer['inp1'] + '",shape = circle];\n'
                if layer['inp1'] not in fea_lst_used:
                    fea_lst_used.append(layer['inp1'])
                    cnt_names.append(layer['inp1'])
                    cnt = cnt + 1

            if layer['inp2'] in fea_lst:
                diag_lines = diag_lines + str(cnt) + ' [label="' + layer['inp2'] + '",shape = circle];\n'
                if layer['inp2'] not in fea_lst_used:
                    fea_lst_used.append(layer['inp2'])
                    cnt_names.append(layer['inp2'])
                    cnt = cnt + 1



        else:
            # architecture
            diag_lines = diag_lines + str(cnt) + ' [label="' + layer['inp1'] + '",shape = box];\n'
            arch_lst.append(layer['inp1'])
            cnt_names.append(layer['inp1'])
            cnt = cnt + 1

            # feature
            if layer['inp2'] in fea_lst:
                diag_lines = diag_lines + str(cnt) + ' [label="' + layer['inp2'] + '",shape = circle];\n'
                if layer['inp2'] not in fea_lst_used:
                    fea_lst_used.append(layer['inp2'])
                    cnt_names.append(layer['inp2'])
                    cnt = cnt + 1

    # Connections across blocks
    lst_conc = fea_lst_used + lab_lst_used

    arch_dict = {}

    for elem in lst_conc:
        arch_dict[elem] = elem

    for model_line in model_arch:
        [layer['out_name'], operation, layer['inp1'], layer['inp2']] = list(re.findall(pattern, model_line)[0])
        if operation == 'compute':
            arch_dict[layer['out_name']] = layer['inp1']
        else:
            arch_dict[layer['out_name']] = layer['out_name']

    output_conn = lst_conc
    [arch_current, output_conn, diag_lines] = create_block_connection(output_conn, model_arch, diag_lines, cnt_names,
                                                                      arch_dict)

    diag_lines = diag_lines + '}'

    # Write the diag file describing the model
    with open(out_diag_file, "w") as text_file:
        text_file.write("%s" % diag_lines)

    # Create image from the diag file
    log_file = config['exp']['out_folder'] + '/log.log'
    cmd = 'blockdiag -Tsvg ' + out_diag_file + ' -o ' + config['exp']['out_folder'] + '/model.svg'
    run_shell(cmd, log_file)


def compute_cw_max(fea_dict):
    cw_left_arr = []
    cw_right_arr = []

    for fea in list(fea_dict.keys()):
        cw_left_arr.append(int(fea_dict[fea][3]))
        cw_right_arr.append(int(fea_dict[fea][4]))

    cw_left_max = max(cw_left_arr)
    cw_right_max = max(cw_right_arr)

    return [cw_left_max, cw_right_max]


def model_init(inp_out_dict, model, config, arch_dict, use_cuda, multi_gpu, to_do):
    nns = {}
    costs = {}

    for layer in model:

        if layer['operation'] == 'compute':

            # computing input dim
            if isinstance(inp_out_dict[layer['inp2']], dict) and 'fea_index_length' in inp_out_dict[layer['inp2']]:
                inp_dim = inp_out_dict[layer['inp2']]['fea_index_length']
            else:
                inp_dim = inp_out_dict[layer['inp2']]

            # import the class
            if arch_dict[layer['inp1']]['arch_class'] == "MLP":
                nn_class = MLP
            elif arch_dict[layer['inp1']]['arch_class'] == "CNN":
                nn_class = CNN
            else:
                raise ValueError

            # add use cuda and todo options
            arch_dict[layer['inp1']]['use_cuda'] = config['exp']['use_cuda']
            arch_dict[layer['inp1']]['to_do'] = config['exp']['to_do']

            # initialize the neural network
            net = nn_class(arch_dict[layer['inp1']], inp_dim)

            if use_cuda:
                net.cuda()
                if multi_gpu:
                    net = nn.DataParallel(net)

            if to_do == 'train':
                net.train()
            else:
                net.eval()

            # addigng nn into the nns dict
            nns[layer['inp1']] = net

            if multi_gpu:
                out_dim = net.module.out_dim
            else:
                out_dim = net.out_dim

            # updating output dim
            inp_out_dict[layer['out_name']] = out_dim

        if layer['operation'] == 'concatenate':
            # TODO fix the -1 and replace with dict lookup by name
            inp_dim1 = inp_out_dict[layer['inp1']][-1]
            inp_dim2 = inp_out_dict[layer['inp2']][-1]

            inp_out_dict[layer['out_name']] = inp_dim1 + inp_dim2

        if layer['operation'] == 'cost_nll':
            costs[layer['out_name']] = nn.NLLLoss()
            inp_out_dict[layer['out_name']] = 1

        if layer['operation'] == 'cost_err':
            inp_out_dict[layer['out_name']] = 1

        if layer['operation'] == 'mult' \
                or layer['operation'] == 'sum' \
                or layer['operation'] == 'mult_constant' \
                or layer['operation'] == 'sum_constant' \
                or layer['operation'] == 'avg' \
                or layer['operation'] == 'mse':
            inp_out_dict[layer['out_name']] = inp_out_dict[layer['inp1']]

    return [nns, costs]


def optimizer_init(nns, config, arch_dict):
    # optimizer init
    optimizers = {}
    for net in nns.keys():

        lr = arch_dict[net]['arch_lr']

        if arch_dict[net]['arch_opt'] == 'sgd':
            optimizers[net] = (optim.SGD(nns[net].parameters(),
                                         lr=lr,
                                         momentum=arch_dict[net]['opt_momentum'],
                                         weight_decay=arch_dict[net]['opt_weight_decay'],
                                         dampening=arch_dict[net]['opt_dampening'],
                                         nesterov=arch_dict[net]['opt_nesterov']))

        if arch_dict[net]['arch_opt'] == 'adam':
            optimizers[net] = (optim.Adam(nns[net].parameters(),
                                          lr=lr,
                                          betas=arch_dict[net]['opt_betas'],
                                          eps=arch_dict[net]['opt_eps'],
                                          weight_decay=arch_dict[net]['opt_weight_decay'],
                                          amsgrad=arch_dict[net]['opt_amsgrad']))

        if arch_dict[net]['arch_opt'] == 'rmsprop':
            optimizers[net] = (optim.RMSprop(nns[net].parameters(),
                                             lr=lr,
                                             momentum=arch_dict[net]['opt_momentum'],
                                             alpha=arch_dict[net]['opt_alpha'],
                                             eps=arch_dict[net]['opt_eps'],
                                             centered=arch_dict[net]['opt_centered'],
                                             weight_decay=arch_dict[net]['opt_weight_decay']))

    return optimizers


def forward_model(fea_dict, lab_dict, arch_dict, model, nns, costs, inp, inp_out_dict, max_len, batch_size, to_do,
                  forward_outs):
    # Forward Step
    outs_dict = {}

    # adding input features to out_dict:
    for fea in fea_dict.keys():
        if len(inp.shape) == 3 and isinstance(fea_dict[fea], dict):
            outs_dict[fea] = inp[:, :, fea_dict[fea]['fea_index_start']:fea_dict[fea]['fea_index_end']]

        if len(inp.shape) == 2 and isinstance(fea_dict[fea], dict):
            outs_dict[fea] = inp[:, fea_dict[fea]['fea_index_start']:fea_dict[fea]['fea_index_end']]

    for layer in model:
        if layer['operation'] == 'compute':

            if isinstance(inp_out_dict[layer['inp2']], dict):  # if it is an input feature

                # Selection of the right feature in the inp tensor
                if len(inp.shape) == 3:
                    inp_dnn = inp[:, :, inp_out_dict[layer['inp2']]['fea_index_start']:
                                        inp_out_dict[layer['inp2']]['fea_index_end']]
                    if not arch_dict[layer['inp1']]['arch_seq_model']:
                        inp_dnn = inp_dnn.view(max_len * batch_size, -1)

                if len(inp.shape) == 2:
                    inp_dnn = inp[:, inp_out_dict[layer['inp2']]['fea_index_start']:
                                     inp_out_dict[layer['inp2']]['fea_index_end']]
                    if arch_dict[layer['inp1']]['arch_seq_model']:
                        inp_dnn = inp_dnn.view(max_len, batch_size, -1)

                outs_dict[layer['out_name']] = nns[layer['inp1']](inp_dnn)


            else:
                if not arch_dict[layer['inp1']]['arch_seq_model'] and len(outs_dict[layer['inp2']].shape) == 3:
                    outs_dict[layer['inp2']] = outs_dict[layer['inp2']].view(max_len * batch_size, -1)

                if arch_dict[layer['inp1']]['arch_seq_model'] and len(outs_dict[layer['inp2']].shape) == 2:
                    outs_dict[layer['inp2']] = outs_dict[layer['inp2']].view(max_len, batch_size, -1)

                outs_dict[layer['out_name']] = nns[layer['inp1']](outs_dict[layer['inp2']])

            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'cost_nll':

            # Put labels in the right format
            if len(inp.shape) == 3:
                lab_dnn = inp[:, :, lab_dict[layer['inp2']]['lab_index']]
            if len(inp.shape) == 2:
                lab_dnn = inp[:, lab_dict[layer['inp2']]['lab_index']]

            lab_dnn = lab_dnn.view(-1).long()

            # put output in the right format
            out = outs_dict[layer['inp1']]

            if len(out.shape) == 3:
                out = out.view(max_len * batch_size, -1)

            assert out.shape[1] >= lab_dnn.max() and lab_dnn.min() >= 0, \
                "lab_dnn max of {} is bigger than shape of output {} or min {} is smaller than 0" \
                    .format(lab_dnn.max().cpu().numpy(), out.shape[1], lab_dnn.min().cpu().numpy())

            if to_do != 'forward':
                outs_dict[layer['out_name']] = costs[layer['out_name']](out, lab_dnn)

        elif layer['operation'] == 'cost_err':

            if len(inp.shape) == 3:
                lab_dnn = inp[:, :, lab_dict[layer['inp2']]['lab_index']]
            if len(inp.shape) == 2:
                lab_dnn = inp[:, lab_dict[layer['inp2']]['lab_index']]

            lab_dnn = lab_dnn.view(-1).long()

            # put output in the right format
            out = outs_dict[layer['inp1']]

            if len(out.shape) == 3:
                out = out.view(max_len * batch_size, -1)

            if to_do != 'forward':
                pred = torch.max(out, dim=1)[1]
                err = torch.mean((pred != lab_dnn).float())
                outs_dict[layer['out_name']] = err
                # print(err)

        elif layer['operation'] == 'concatenate':
            dim_conc = len(outs_dict[layer['inp1']].shape) - 1
            outs_dict[layer['out_name']] = torch.cat((outs_dict[layer['inp1']], outs_dict[layer['inp2']]),
                                                     dim_conc)  # check concat axis
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'mult':
            outs_dict[layer['out_name']] = outs_dict[layer['inp1']] * outs_dict[layer['inp2']]
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'sum':
            outs_dict[layer['out_name']] = outs_dict[layer['inp1']] + outs_dict[layer['inp2']]
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'mult_constant':
            outs_dict[layer['out_name']] = outs_dict[layer['inp1']] * float(layer['inp2'])
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'sum_constant':
            outs_dict[layer['out_name']] = outs_dict[layer['inp1']] + float(layer['inp2'])
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'avg':
            outs_dict[layer['out_name']] = (outs_dict[layer['inp1']] + outs_dict[layer['inp2']]) / 2
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break

        elif layer['operation'] == 'mse':
            outs_dict[layer['out_name']] = torch.mean((outs_dict[layer['inp1']] - outs_dict[layer['inp2']]) ** 2)
            if to_do == 'forward' and layer['out_name'] == forward_outs[-1]:
                break
        else:
            print("WARING!!!!!!!!!")

    return outs_dict


def dump_epoch_results(res_file_path, ep, tr_data_lst, tr_loss_tot, tr_error_tot, tot_time, valid_data_lst,
                       valid_peformance_dict, lr, N_ep):
    #
    # Default terminal line size is 80 characters, try new dispositions to fit this limit
    #

    res_file = open(res_file_path, "a")
    res_file.write('ep=%s tr=%s loss=%s err=%s ' % (
        format(ep, "03d"), tr_data_lst, format(tr_loss_tot / len(tr_data_lst), "0.3f"),
        format(tr_error_tot / len(tr_data_lst), "0.3f")))
    print(' ')
    print(('----- Summary epoch %s / %s' % (format(ep, "03d"), format(N_ep - 1, "03d"))))
    print(('Training on %s' % (tr_data_lst)))
    print(('Loss = %s | err = %s ' % (
        format(tr_loss_tot / len(tr_data_lst), "0.3f"), format(tr_error_tot / len(tr_data_lst), "0.3f"))))
    print('-----')
    for valid_data in valid_data_lst:
        res_file.write('valid=%s loss=%s err=%s ' % (valid_data, format(valid_peformance_dict[valid_data][0], "0.3f"),
                                                     format(valid_peformance_dict[valid_data][1], "0.3f")))
        print(('Validating on %s' % (valid_data)))
        print(('Loss = %s | err = %s ' % (
            format(valid_peformance_dict[valid_data][0], "0.3f"),
            format(valid_peformance_dict[valid_data][1], "0.3f"))))

    print('-----')
    for lr_arch in list(lr.keys()):
        res_file.write('lr_%s=%f ' % (lr_arch, lr[lr_arch]))
        print(('Learning rate on %s = %f ' % (lr_arch, lr[lr_arch])))
    print('-----')
    res_file.write('time(s)=%i\n' % (int(tot_time)))
    print(('Elapsed time (s) = %i\n' % (int(tot_time))))
    print(' ')
    res_file.close()


def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count == total - 1:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, 100, '%', status))
        sys.stdout.write("\n")
    else:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def export_loss_acc_to_txt(out_folder, N_ep, val_lst):
    if not os.path.exists(out_folder + '/generated_outputs'):
        os.makedirs(out_folder + '/generated_outputs')

    nb_val = len(val_lst)
    res = open(out_folder + '/res.res', 'r').readlines()

    tr_loss = []
    tr_acc = []
    val_loss = np.ndarray((nb_val, N_ep))
    val_acc = np.ndarray((nb_val, N_ep))

    line_cpt = 0
    for i in range(N_ep):
        splitted = res[i].split(' ')

        # Getting uniq training loss and acc
        tr_loss.append(float(splitted[2].split('=')[1]))
        tr_acc.append(1 - float(splitted[3].split('=')[1]))

        # Getting multiple or uniq val loss and acc
        # +5 to avoird the 6 first columns of the res.res file
        for i in range(nb_val):
            val_loss[i][line_cpt] = float(splitted[(i * 3) + 5].split('=')[1])
            val_acc[i][line_cpt] = 1 - float(splitted[(i * 3) + 6].split('=')[1])

        line_cpt += 1

    # Saving to files
    np.savetxt(out_folder + '/generated_outputs/tr_loss.txt', np.asarray(tr_loss), '%0.3f', delimiter=',')
    np.savetxt(out_folder + '/generated_outputs/tr_acc.txt', np.asarray(tr_acc), '%0.3f', delimiter=',')

    for i in range(nb_val):
        np.savetxt(out_folder + '/generated_outputs/val_' + str(i) + '_loss.txt', val_loss[i], '%0.5f', delimiter=',')
        np.savetxt(out_folder + '/generated_outputs/val_' + str(i) + '_acc.txt', val_acc[i], '%0.5f', delimiter=',')


def create_curves(out_folder, N_ep, val_lst):
    print(' ')
    print('-----')
    print('Generating output files and plots ... ')
    export_loss_acc_to_txt(out_folder, N_ep, val_lst)

    if not os.path.exists(out_folder + '/generated_outputs'):
        raise RuntimeError('accOR: No results generated please call export_loss_err_to_txt() before')

    nb_epoch = len(open(out_folder + '/generated_outputs/tr_loss.txt', 'r').readlines())
    x = np.arange(nb_epoch)
    nb_val = len(val_lst)

    # Loading train Loss and acc
    tr_loss = np.loadtxt(out_folder + '/generated_outputs/tr_loss.txt')
    tr_acc = np.loadtxt(out_folder + '/generated_outputs/tr_acc.txt')

    # Loading val loss and acc
    val_loss = []
    val_acc = []
    for i in range(nb_val):
        val_loss.append(np.loadtxt(out_folder + '/generated_outputs/val_' + str(i) + '_loss.txt'))
        val_acc.append(np.loadtxt(out_folder + '/generated_outputs/val_' + str(i) + '_acc.txt'))

    #
    # LOSS PLOT
    #

    # Getting maximum values
    max_loss = np.amax(tr_loss)
    for i in range(nb_val):
        if np.amax(val_loss[i]) > max_loss:
            max_loss = np.amax(val_loss[i])

    # Plot train loss and acc
    plt.plot(x, tr_loss, label="train_loss")

    # Plot val loss and acc
    for i in range(nb_val):
        plt.plot(x, val_loss[i], label='val_' + str(i) + '_loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Evolution of the loss function')
    plt.axis([0, nb_epoch - 1, 0, max_loss + 1])
    plt.legend()
    plt.savefig(out_folder + '/generated_outputs/loss.png')

    # Clear plot
    plt.gcf().clear()

    #
    # ACC PLOT
    #

    # Plot train loss and acc
    plt.plot(x, tr_acc, label="train_acc")

    # Plot val loss and acc
    for i in range(nb_val):
        plt.plot(x, val_acc[i], label='val_' + str(i) + '_acc')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Evolution of the accuracy')
    plt.axis([0, nb_epoch - 1, 0, 1])
    plt.legend()
    plt.savefig(out_folder + '/generated_outputs/acc.png')

    print('OK')
