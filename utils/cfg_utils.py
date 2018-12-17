import configparser
import sys
import os.path
import re
from distutils.util import strtobool


def check_field(inp, type_inp, field):
    valid_field = True

    if inp == '' and field != 'cmd':
        raise ValueError("ERROR: The the field  \"%s\" of the config file is empty! \n" % (field))
        valid_field = False

    if type_inp == 'path':
        if not (os.path.isfile(inp)) and not (os.path.isdir(inp)) and inp != 'none':
            raise ValueError(
                "ERROR: The path \"%s\" specified in the field  \"%s\" of the config file does not exists! \n" % (
                    inp, field))
            valid_field = False

    if '{' and '}' in type_inp:
        arg_list = type_inp[1:-1].split(',')
        if inp not in arg_list:
            raise ValueError("ERROR: The field \"%s\" can only contain %s  arguments \n" % (field, arg_list))
            valid_field = False

    if 'int(' in type_inp:
        try:
            int(inp)
        except ValueError:
            raise ValueError("ERROR: The field \"%s\" can only contain an integer (got \"%s\") \n" % (field, inp))
            valid_field = False

        # Check if the value if within the expected range
        lower_bound = type_inp.split(',')[0][4:]
        upper_bound = type_inp.split(',')[1][:-1]

        if lower_bound != "-inf":
            if int(inp) < int(lower_bound):
                raise ValueError(
                    "ERROR: The field \"%s\" can only contain an integer greater than %s (got \"%s\") \n" % (
                        field, lower_bound, inp))
                valid_field = False

        if upper_bound != "inf":
            if int(inp) > int(upper_bound):
                raise ValueError(
                    "ERROR: The field \"%s\" can only contain an integer smaller than %s (got \"%s\") \n" % (
                        field, upper_bound, inp))
                valid_field = False

    if 'float(' in type_inp:
        try:
            float(inp)
        except ValueError:
            raise ValueError("ERROR: The field \"%s\" can only contain a float (got \"%s\") \n" % (field, inp))
            valid_field = False

        # Check if the value if within the expected range
        lower_bound = type_inp.split(',')[0][6:]
        upper_bound = type_inp.split(',')[1][:-1]

        if lower_bound != "-inf":
            if float(inp) < float(lower_bound):
                raise ValueError("ERROR: The field \"%s\" can only contain a float greater than %s (got \"%s\") \n" % (
                    field, lower_bound, inp))
                valid_field = False

        if upper_bound != "inf":
            if float(inp) > float(upper_bound):
                raise ValueError("ERROR: The field \"%s\" can only contain a float smaller than %s (got \"%s\") \n" % (
                    field, upper_bound, inp))
                valid_field = False

    if type_inp == 'bool':
        lst = {'True', 'true', '1', 'False', 'false', '0'}
        if not (inp in lst):
            raise ValueError("ERROR: The field \"%s\" can only contain a boolean (got \"%s\") \n" % (field, inp))
            valid_field = False

    if 'int_list(' in type_inp:
        lst = inp.split(',')
        try:
            list(map(int, lst))
        except ValueError:
            raise ValueError(
                "ERROR: The field \"%s\" can only contain a list of integer (got \"%s\") \n" % (field, inp))
            valid_field = False

        # Check if the value if within the expected range
        lower_bound = type_inp.split(',')[0][9:]
        upper_bound = type_inp.split(',')[1][:-1]

        for elem in lst:

            if lower_bound != "-inf":
                if int(elem) < int(lower_bound):
                    raise ValueError(
                        "ERROR: The field \"%s\" can only contain an integer greater than %s (got \"%s\") \n" % (
                            field, lower_bound, elem))
                    valid_field = False

            if upper_bound != "inf":
                if int(elem) > int(upper_bound):
                    raise ValueError(
                        "ERROR: The field \"%s\" can only contain an integer smaller than %s (got \"%s\") \n" % (
                            field, upper_bound, elem))
                    valid_field = False

    if 'float_list(' in type_inp:
        lst = inp.split(',')
        try:
            list(map(float, lst))
        except ValueError:
            raise ValueError("ERROR: The field \"%s\" can only contain a list of floats (got \"%s\") \n" % (field, inp))
            valid_field = False

        # Check if the value if within the expected range
        lower_bound = type_inp.split(',')[0][11:]
        upper_bound = type_inp.split(',')[1][:-1]

        for elem in lst:

            if lower_bound != "-inf":
                if float(elem) < float(lower_bound):
                    raise ValueError(
                        "ERROR: The field \"%s\" can only contain a float greater than %s (got \"%s\") \n" % (
                            field, lower_bound, elem))
                    valid_field = False

            if upper_bound != "inf":
                if float(elem) > float(upper_bound):
                    raise ValueError(
                        "ERROR: The field \"%s\" can only contain a float smaller than %s (got \"%s\") \n" % (
                            field, upper_bound, elem))
                    valid_field = False

    if type_inp == 'bool_list':
        lst = {'True', 'true', '1', 'False', 'false', '0'}
        inps = inp.split(',')
        for elem in inps:
            if not (elem in lst):
                raise ValueError(
                    "ERROR: The field \"%s\" can only contain a list of boolean (got \"%s\") \n" % (field, inp))
                valid_field = False

    return valid_field


def get_all_archs(config):
    arch_lst = []
    for sec in config.sections():
        if 'architecture' in sec:
            arch_lst.append(sec)
    return arch_lst


def expand_section(config_proto, config):
    # expands config_proto with fields in prototype files
    name_data = []
    name_arch = []
    for sec in config.sections():
        if 'dataset' in sec:
            config_proto.add_section(sec)
            config_proto[sec] = config_proto['dataset']
            name_data.append(config[sec]['data_name'])

        if 'architecture' in sec:
            name_arch.append(config[sec]['arch_name'])
            config_proto.add_section(sec)
            config_proto[sec] = config_proto['architecture']
            proto_file = config[sec]['arch_proto']

            # Reading proto file (architecture)
            config_arch = configparser.ConfigParser()
            config_arch.read(proto_file)

            # Reading proto options
            fields_arch = list(dict(config_arch.items('proto')).keys())
            fields_arch_type = list(dict(config_arch.items('proto')).values())

            for i in range(len(fields_arch)):
                config_proto.set(sec, fields_arch[i], fields_arch_type[i])

            # Reading proto file (architecture_optimizer)
            opt_type = config[sec]['arch_opt']
            if opt_type == 'sgd':
                proto_file = 'proto/sgd.proto'

            if opt_type == 'rmsprop':
                proto_file = 'proto/rmsprop.proto'

            if opt_type == 'adam':
                proto_file = 'proto/adam.proto'

            config_arch = configparser.ConfigParser()
            config_arch.read(proto_file)

            # Reading proto options
            fields_arch = list(dict(config_arch.items('proto')).keys())
            fields_arch_type = list(dict(config_arch.items('proto')).values())

            for i in range(len(fields_arch)):
                config_proto.set(sec, fields_arch[i], fields_arch_type[i])

    config_proto.remove_section('dataset')
    config_proto.remove_section('architecture')

    return [config_proto, name_data, name_arch]


def expand_section_proto(config_proto, config):
    # Read config proto file
    config_proto_optim_file = config['optimization']['opt_proto']
    config_proto_optim = configparser.ConfigParser()
    config_proto_optim.read(config_proto_optim_file)
    for optim_par in list(config_proto_optim['proto']):
        config_proto.set('optimization', optim_par, config_proto_optim['proto'][optim_par])


def check_cfg_fields(config_proto, config, cfg_file):
    # Check mandatory sections and fields
    sec_parse = True

    for sec in config_proto.sections():

        if any(sec in s for s in config.sections()):

            # Check fields
            for field in list(dict(config_proto.items(sec)).keys()):

                if not (field in config[sec]):
                    raise ValueError(
                        "ERROR: The confg file %s does not contain the field \"%s=\" in section  \"[%s]\" (mandatory)!\n" % (
                            cfg_file, field, sec))
                    sec_parse = False
                else:
                    field_type = config_proto[sec][field]
                    if not (check_field(config[sec][field], field_type, field)):
                        sec_parse = False



        # If a mandatory section doesn't exist...
        else:
            raise ValueError(
                "ERROR: The confg file %s does not contain \"[%s]\" section (mandatory)!\n" % (cfg_file, sec))
            sec_parse = False

    if sec_parse == False:
        raise ValueError("ERROR: Revise the confg file %s \n" % (cfg_file))
    return sec_parse


def check_consistency_with_proto(cfg_file, cfg_file_proto):
    sec_parse = True

    # Check if cfg file exists
    try:
        open(cfg_file, 'r')
    except IOError:
        raise ValueError("ERROR: The confg file %s does not exist!\n" % (cfg_file))

    # Check if cfg proto  file exists
    try:
        open(cfg_file_proto, 'r')
    except IOError:
        raise ValueError("ERROR: The confg file %s does not exist!\n" % (cfg_file_proto))

        # Parser Initialization
    config = configparser.ConfigParser()

    # Reading the cfg file
    config.read(cfg_file)

    # Reading proto cfg file
    config_proto = configparser.ConfigParser()
    config_proto.read(cfg_file_proto)

    # Adding the multiple entries in data and architecture sections
    [config_proto, name_data, name_arch] = expand_section(config_proto, config)

    # Check mandatory sections and fields
    sec_parse = check_cfg_fields(config_proto, config, cfg_file)

    if sec_parse == False:
        return

    return [config_proto, name_data, name_arch]


def check_cfg(cfg_file, config, cfg_file_proto):
    # Check consistency between cfg_file and cfg_file_proto
    [config_proto, name_data, name_arch] = check_consistency_with_proto(cfg_file, cfg_file_proto)

    # check consistency between [data_use] vs [data*]
    sec_parse = True
    data_use_with = []
    for data in list(dict(config.items('data_use')).values()):
        data_use_with.append(data.split(','))

    data_use_with = sum(data_use_with, [])

    if not (set(data_use_with).issubset(name_data)):
        raise ValueError("ERROR: in [data_use] you are using a dataset not specified in [dataset*] %s \n" % (cfg_file))
        sec_parse = False

    # Parse fea and lab  fields in datasets*
    cnt = 0
    fea_names_lst = []
    lab_names_lst = []
    for data in name_data:

        [fea_names, fea_lsts, fea_opts, cws_left, cws_right] = parse_fea_field(
            config[cfg_item2sec(config, 'data_name', data)]['fea'])
        [lab_names, lab_folders, lab_opts] = parse_lab_field(config[cfg_item2sec(config, 'data_name', data)]['lab'])

        fea_names_lst.append(sorted(fea_names))
        lab_names_lst.append(sorted(lab_names))

        if cnt > 0:
            if fea_names_lst[cnt - 1] != fea_names_lst[cnt]:
                raise ValueError("features name (fea_name) must be the same of all the datasets! \n")
                sec_parse = False

            if lab_names_lst[cnt - 1] != lab_names_lst[cnt]:
                raise ValueError("labels name (lab_name) must be the same of all the datasets! \n")
                sec_parse = False

        cnt = cnt + 1

    # Create the output folder
    out_folder = config['exp']['out_folder']

    if not os.path.exists(out_folder) or not (os.path.exists(out_folder + '/exp_files')):
        os.makedirs(out_folder + '/exp_files')

    # Parsing forward field
    model = config['model']['model']
    possible_outs = list(re.findall('(.*)=', model.replace(' ', '')))
    forward_out_lst = config['forward']['forward_out'].split(',')
    forward_norm_lst = config['forward']['normalize_with_counts_from'].split(',')
    forward_norm_bool_lst = config['forward']['normalize_posteriors'].split(',')

    lab_lst = list(re.findall('lab_name=(.*)\n', config['dataset1']['lab'].replace(' ', '')))
    lab_folders = list(re.findall('lab_folder=(.*)\n', config['dataset1']['lab'].replace(' ', '')))
    N_out_lab = ['none'] * len(lab_lst)

    for i in range(len(forward_out_lst)):
        if forward_out_lst[i] not in possible_outs:
            raise ValueError(
                'ERROR: the output \"%s\" in the section \"forwad_out\" is not defined in section model)\n' % (
                    forward_out_lst[i]))

        if strtobool(forward_norm_bool_lst[i]):

            if forward_norm_lst[i] not in lab_lst:
                if not os.path.exists(forward_norm_lst[i]):
                    raise ValueError(
                        'ERROR: the count_file \"%s\" in the section \"forwad_out\" is does not exist)\n' % (
                            forward_norm_lst[i]))

                else:
                    # Check if the specified file is in the right format
                    f = open(forward_norm_lst[i], "r")
                    cnts = f.read()
                    if not (bool(re.match("(.*)\[(.*)\]", cnts))):
                        raise ValueError(
                            'ERROR: the count_file \"%s\" in the section \"forwad_out\" is not in the right format)\n' % (
                                forward_norm_lst[i]))


            else:
                # Try to automatically retrieve the config file
                if "ali-to-pdf" in lab_opts[lab_lst.index(forward_norm_lst[i])]:
                    log_file = config['exp']['out_folder'] + '/log.log'
                    folder_lab_count = lab_folders[lab_lst.index(forward_norm_lst[i])]
                    cmd = "hmm-info " + folder_lab_count + "/final.mdl | awk '/pdfs/{print $4}'"
                    output = run_shell(cmd, log_file)
                    N_out = int(output.decode().rstrip())
                    N_out_lab[lab_lst.index(forward_norm_lst[i])] = N_out
                    count_file_path = out_folder + '/exp_files/forward_' + forward_out_lst[i] + '_' + forward_norm_lst[
                        i] + '.count'
                    cmd = "analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim=" + str(
                        N_out) + " \"ark:ali-to-pdf " + folder_lab_count + "/final.mdl \\\"ark:gunzip -c " + folder_lab_count + "/ali.*.gz |\\\" ark:- |\" " + count_file_path
                    run_shell(cmd, log_file)
                    forward_norm_lst[i] = count_file_path

                else:
                    raise ValueError(
                        'ERROR: Not able to automatically retrieve count file for the label \"%s\". Please add a valid count file path in \"normalize_with_counts_from\" or set normalize_posteriors=False \n' % (
                            forward_norm_lst[i]))

    # Update the config file with the count_file paths
    config['forward']['normalize_with_counts_from'] = ",".join(forward_norm_lst)

    # When possible replace the pattern "N_out_lab*" with the detected number of output

    for sec in config.sections():
        for field in list(config[sec]):
            for i in range(len(lab_lst)):
                pattern = 'N_out_' + lab_lst[i]

                if pattern in config[sec][field]:
                    if N_out_lab[i] != 'none':
                        config[sec][field] = config[sec][field].replace(pattern, str(N_out_lab[i]))
                    else:
                        raise ValueError(
                            'ERROR: Cannot automatically retrieve the number of output in %s. Plese, add manually the number of outputs \n' % (
                                pattern))

    # Check the model field
    parse_model_field(cfg_file)

    # Create block diagram picture of the model
    create_block_diagram(cfg_file)

    if sec_parse == False:
        return

    return [config, name_data, name_arch]


def cfg_item2sec(config, field, value):
    for sec in config.sections():
        if field in list(dict(config.items(sec)).keys()):
            if value in list(dict(config.items(sec)).values()):
                return sec

    raise ValueError("ERROR: %s=%s not found in config file \n" % (field, value))

    return -1


def parse_fea_field(fea):
    # Adding the required fields into a list
    fea_names = []
    fea_lsts = []
    fea_opts = []
    cws_left = []
    cws_right = []

    for line in fea.split('\n'):

        line = re.sub(' +', ' ', line)

        if 'fea_name=' in line:
            fea_names.append(line.split('=')[1])

        if 'fea_lst=' in line:
            fea_lsts.append(line.split('=')[1])

        if 'fea_opts=' in line:
            fea_opts.append(line.split('fea_opts=')[1])

        if 'cw_left=' in line:
            cws_left.append(line.split('=')[1])
            if not (check_field(line.split('=')[1], 'int(0,inf)', 'cw_left')):
                raise RuntimeError

        if 'cw_right=' in line:
            cws_right.append(line.split('=')[1])
            if not (check_field(line.split('=')[1], 'int(0,inf)', 'cw_right')):
                raise RuntimeError

                # Check features names
    if not (sorted(fea_names) == sorted(list(set(fea_names)))):
        raise ValueError('ERROR fea_names must be different! (got %s)' % (fea_names))

    snt_lst = []
    cnt = 0

    # Check consistency of feature lists
    for fea_lst in fea_lsts:
        if not (os.path.isfile(fea_lst)):
            raise ValueError(
                "ERROR: The path \"%s\" specified in the field  \"fea_lst\" of the config file does not exists! \n" % (
                    fea_lst))
        else:
            snts = sorted([line.rstrip('\n').split(' ')[0] for line in open(fea_lst)])
            snt_lst.append(snts)
            # Check if all the sentences are present in all the list files
            if cnt > 0:
                if snt_lst[cnt - 1] != snt_lst[cnt]:
                    raise ValueError(
                        "ERROR: the files %s in fea_lst contain a different set of sentences! \n" % (fea_lst))
            cnt = cnt + 1
    return [fea_names, fea_lsts, fea_opts, cws_left, cws_right]


def parse_lab_field(lab):
    # Adding the required fields into a list
    lab_names = []
    lab_folders = []
    lab_opts = []

    for line in lab.split('\n'):

        line = re.sub(' +', ' ', line)

        if 'lab_name=' in line:
            lab_names.append(line.split('=')[1])

        if 'lab_folder=' in line:
            lab_folders.append(line.split('=')[1])

        if 'lab_opts=' in line:
            lab_opts.append(line.split('lab_opts=')[1])

            # Check features names
    if not (sorted(lab_names) == sorted(list(set(lab_names)))):
        raise ValueError('ERROR lab_names must be different! (got %s)' % (lab_names))

    # Check consistency of feature lists
    for lab_fold in lab_folders:
        if not (os.path.isdir(lab_fold)):
            raise ValueError(
                "ERROR: The path \"%s\" specified in the field  \"lab_folder\" of the config file does not exists! \n" % (
                    lab_fold))

    return [lab_names, lab_folders, lab_opts]


def list_fea_lab_arch(config):  # cancel
    model = config['model']['model'].split('\n')
    fea_lst = list(re.findall('fea_name=(.*)\n', config['data_chunk']['fea'].replace(' ', '')))
    lab_lst = list(re.findall('lab_name=(.*)\n', config['data_chunk']['lab'].replace(' ', '')))

    fea_lst_used = []
    lab_lst_used = []
    arch_lst_used = []

    fea_dict_used = {}
    lab_dict_used = {}
    arch_dict_used = {}

    fea_lst_used_name = []
    lab_lst_used_name = []
    arch_lst_used_name = []

    fea_field = config['data_chunk']['fea']
    lab_field = config['data_chunk']['lab']

    pattern = '(.*)=(.*)\((.*),(.*)\)'

    for line in model:
        [out_name, operation, inp1, inp2] = list(re.findall(pattern, line)[0])

        if inp1 in fea_lst and inp1 not in fea_lst_used_name:
            pattern_fea = "fea_name=" + inp1 + "\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            fea_lst_used.append((inp1 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(','))
            fea_dict_used[inp1] = (inp1 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(',')

            fea_lst_used_name.append(inp1)
        if inp2 in fea_lst and inp2 not in fea_lst_used_name:
            pattern_fea = "fea_name=" + inp2 + "\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            fea_lst_used.append((inp2 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(','))
            fea_dict_used[inp2] = (inp2 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(',')

            fea_lst_used_name.append(inp2)
        if inp1 in lab_lst and inp1 not in lab_lst_used_name:
            pattern_lab = "lab_name=" + inp1 + "\nlab_folder=(.*)\nlab_opts=(.*)"
            lab_lst_used.append((inp1 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(','))
            lab_dict_used[inp1] = (inp1 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(',')

            lab_lst_used_name.append(inp1)

        if inp2 in lab_lst and inp2 not in lab_lst_used_name:
            pattern_lab = "lab_name=" + inp2 + "\nlab_folder=(.*)\nlab_opts=(.*)"
            lab_lst_used.append((inp2 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(','))
            lab_dict_used[inp2] = (inp2 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(',')

            lab_lst_used_name.append(inp2)

        if operation == 'compute' and inp1 not in arch_lst_used_name:
            arch_id = cfg_item2sec(config, 'arch_name', inp1)
            arch_seq_model = strtobool(config[arch_id]['arch_seq_model'])
            arch_lst_used.append([arch_id, inp1, arch_seq_model])
            arch_dict_used[inp1] = [arch_id, inp1, arch_seq_model]

            arch_lst_used_name.append(inp1)

    # convert to unicode (for python 2)
    for i in range(len(fea_lst_used)):
        fea_lst_used[i] = list(map(str, fea_lst_used[i]))

    for i in range(len(lab_lst_used)):
        lab_lst_used[i] = list(map(str, lab_lst_used[i]))

    for i in range(len(arch_lst_used)):
        arch_lst_used[i] = list(map(str, arch_lst_used[i]))

    return [fea_lst_used, lab_lst_used, arch_lst_used]


def dict_fea_lab_arch(config):
    model = config['model']['model'].split('\n')
    fea_lst = list(re.findall('fea_name=(.*)\n', config['data_chunk']['fea'].replace(' ', '')))
    lab_lst = list(re.findall('lab_name=(.*)\n', config['data_chunk']['lab'].replace(' ', '')))

    fea_lst_used = []
    lab_lst_used = []
    arch_lst_used = []

    fea_dict_used = {}
    lab_dict_used = {}
    arch_dict_used = {}

    fea_lst_used_name = []
    lab_lst_used_name = []
    arch_lst_used_name = []

    fea_field = config['data_chunk']['fea']
    lab_field = config['data_chunk']['lab']

    pattern = '(.*)=(.*)\((.*),(.*)\)'

    for line in model:
        [out_name, operation, inp1, inp2] = list(re.findall(pattern, line)[0])

        if inp1 in fea_lst and inp1 not in fea_lst_used_name:
            pattern_fea = "fea_name=" + inp1 + "\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            if sys.version_info[0] == 2:
                fea_lst_used.append(
                    (inp1 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).encode('utf8').split(','))
                fea_dict_used[inp1] = (inp1 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).encode(
                    'utf8').split(',')
            else:
                fea_lst_used.append((inp1 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(','))
                fea_dict_used[inp1] = (inp1 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(',')

            fea_lst_used_name.append(inp1)

        if inp2 in fea_lst and inp2 not in fea_lst_used_name:
            pattern_fea = "fea_name=" + inp2 + "\nfea_lst=(.*)\nfea_opts=(.*)\ncw_left=(.*)\ncw_right=(.*)"
            if sys.version_info[0] == 2:
                fea_lst_used.append(
                    (inp2 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).encode('utf8').split(','))
                fea_dict_used[inp2] = (inp2 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).encode(
                    'utf8').split(',')
            else:
                fea_lst_used.append((inp2 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(','))
                fea_dict_used[inp2] = (inp2 + "," + ",".join(list(re.findall(pattern_fea, fea_field)[0]))).split(',')

            fea_lst_used_name.append(inp2)
        if inp1 in lab_lst and inp1 not in lab_lst_used_name:
            pattern_lab = "lab_name=" + inp1 + "\nlab_folder=(.*)\nlab_opts=(.*)"

            if sys.version_info[0] == 2:
                lab_lst_used.append(
                    (inp1 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).encode('utf8').split(','))
                lab_dict_used[inp1] = (inp1 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).encode(
                    'utf8').split(',')
            else:
                lab_lst_used.append((inp1 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(','))
                lab_dict_used[inp1] = (inp1 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(',')

            lab_lst_used_name.append(inp1)

        if inp2 in lab_lst and inp2 not in lab_lst_used_name:
            pattern_lab = "lab_name=" + inp2 + "\nlab_folder=(.*)\nlab_opts=(.*)"

            if sys.version_info[0] == 2:
                lab_lst_used.append(
                    (inp2 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).encode('utf8').split(','))
                lab_dict_used[inp2] = (inp2 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).encode(
                    'utf8').split(',')
            else:
                lab_lst_used.append((inp2 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(','))
                lab_dict_used[inp2] = (inp2 + "," + ",".join(list(re.findall(pattern_lab, lab_field)[0]))).split(',')

            lab_lst_used_name.append(inp2)

        if operation == 'compute' and inp1 not in arch_lst_used_name:
            arch_id = cfg_item2sec(config, 'arch_name', inp1)
            arch_seq_model = strtobool(config[arch_id]['arch_seq_model'])
            arch_lst_used.append([arch_id, inp1, arch_seq_model])
            arch_dict_used[inp1] = [arch_id, inp1, arch_seq_model]

            arch_lst_used_name.append(inp1)

    # convert to unicode (for python 2)
    for i in range(len(fea_lst_used)):
        fea_lst_used[i] = list(map(str, fea_lst_used[i]))

    for i in range(len(lab_lst_used)):
        lab_lst_used[i] = list(map(str, lab_lst_used[i]))

    for i in range(len(arch_lst_used)):
        arch_lst_used[i] = list(map(str, arch_lst_used[i]))
    """
    fea_dict_used:
    {
     "mfcc": [
      "mfcc",
      "exp/TIMIT_MLP_basic/exp_files/train_TIMIT_tr_ep000_ck00_mfcc.lst",
      "apply-cmvn --utt2spk=ark:/mnt/data/libs/kaldi/egs/timit/s5/data/train/utt2spk  ark:/mnt/data/libs/kaldi/egs/timit/s5/mfcc/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |",
      "5",
      "5"
     ]
    }
    lab_dict_used:
    {
     "lab_cd": [
      "lab_cd",
      "/mnt/data/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali",
      "ali-to-pdf"
     ]
    }
    arch_dict_used:
    {
     "MLP_layers1": [
      "architecture1",
      "MLP_layers1",
      0
     ]
    }
    """
    return [fea_dict_used, lab_dict_used, arch_dict_used]


def is_sequential(config, arch_lst):  # To cancel
    seq_model = False

    for [arch_id, arch_name, arch_seq] in arch_lst:
        if strtobool(config[arch_id]['arch_seq_model']):
            seq_model = True
            break
    return seq_model


def is_sequential_dict(config, arch_dict):
    """['arch_seq_model'] in arch is true or false"""
    seq_model = False

    for arch in list(arch_dict.keys()):
        arch_id = arch_dict[arch][0]
        if strtobool(config[arch_id]['arch_seq_model']):
            seq_model = True
            break
    return seq_model
