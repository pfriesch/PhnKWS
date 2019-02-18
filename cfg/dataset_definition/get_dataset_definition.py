from utils.utils import run_shell, read_json


def get_lab_count(label_opts, num_label, folder_lab_count):
    cmd = f'analyze-counts --print-args=False --verbose=0 --binary=false' \
          + f' --counts-dim={num_label} "ark:{label_opts}' \
          + f' {folder_lab_count}/final.mdl \\"ark:gunzip -c ' \
          + f'{folder_lab_count}/ali.*.gz |\\" ark:- |" -'
    # + f'{folder_lab_count}/ali.*.gz |\\" ark:- |" {count_file_path}'
    # count_file_path = "tmp.count" #TODO save in file instead

    lab_count = run_shell(cmd)

    lab_count = lab_count.strip().strip('[]').strip()
    lab_count = [float(v) for v in lab_count.split()]
    return lab_count


def get_dataset_definition(dataset_name, train_with):
    if dataset_name == "librispeech":
        dataset_config_path = "cfg/dataset_definition/librispeech.json"
    elif dataset_name == "TIMIT":
        dataset_config_path = "cfg/dataset_definition/TIMIT.json"
    else:
        raise NotImplementedError(dataset_name)

    dataset_definition = read_json(dataset_config_path)
    for label in dataset_definition['data_info']['labels']:
        label_info = dataset_definition['data_info']['labels'][label]
        if label_info['num_lab'] is None:
            if label == "lab_cd":

                folder_lab_count = dataset_definition['datasets'][train_with] \
                    ['labels'][label]['label_folder']
                hmm_info = run_shell(f"hmm-info {folder_lab_count}/final.mdl")
                label_info['num_lab'] = int(hmm_info.split("\n")[1].rsplit(" ", 1)[1])

                label_info['lab_count'] = get_lab_count(
                    label_opts=dataset_definition['datasets'][train_with] \
                        ['labels'][label]['label_opts'],
                    num_label=label_info["num_lab"],
                    folder_lab_count=folder_lab_count)

            elif label == "lab_mono" or label == "lab_phn":
                folder_lab_count = dataset_definition['datasets'][train_with] \
                    ['labels'][label]['label_folder']
                hmm_info = run_shell(f"hmm-info {folder_lab_count}/final.mdl")
                label_info['num_lab'] = int(hmm_info.split("\n")[0].rsplit(" ", 1)[1])

                label_info['lab_count'] = get_lab_count(
                    label_opts=dataset_definition['datasets'][train_with] \
                        ['labels'][label]['label_opts'],
                    num_label=label_info["num_lab"],
                    folder_lab_count=folder_lab_count)

            else:
                raise NotImplementedError(label)

    return dataset_definition
