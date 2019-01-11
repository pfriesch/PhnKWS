import os
import re

import subprocess
import warnings

import collections


def recursive_update(_dict, _update_dict):
    for k, v in _update_dict.items():
        if isinstance(v, collections.Mapping):
            _dict[k] = recursive_update(_dict.get(k, {}), v)
        else:
            _dict[k] = v
    return _dict


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def folder_to_checkpoint(folder):
    checkpoint_folder_path = os.path.join(folder, 'checkpoints')
    checkpoints = os.listdir(checkpoint_folder_path)
    checkpoints = [(re.findall("checkpoint-epoch(.*)\.pth", cp), cp) for cp in checkpoints]

    resume_path = max(checkpoints, key=lambda x: x[0])[1]
    return os.path.join(checkpoint_folder_path, resume_path)


def code_versioning():
    try:
        # todo make sure exp folder is in gitignore
        branches = subprocess.check_output(['git', 'branch']).decode("utf-8").split("\n")
        current_branch = [b for b in branches if b.startswith("*")][0].split(" ")[1]
        assert current_branch == 'runs_log'

        subprocess.check_output(['git', 'add', '--all'])
        _ret_val = subprocess.check_output(['git', 'commit', '-am', 'update'])

        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
    except Exception as e:
        print(e)
        warnings.warn(e)
