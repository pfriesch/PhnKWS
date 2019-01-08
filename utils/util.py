import os
import re

import subprocess


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def folder_to_checkpoint(folder):
    checkpoint_folder_path = os.path.join(folder, 'checkpoints')
    checkpoints = os.listdir(checkpoint_folder_path)
    checkpoints = [(re.findall("checkpoint-epoch(.*)\.pth", cp), cp) for cp in checkpoints]

    resume_path = max(checkpoints, key=_[0])[1]
    return os.path.join(checkpoint_folder_path, resume_path)


def code_versioning():
    #todo make sure exp folder is in gitignore
    branches = subprocess.check_output(['git', 'branch']).decode("utf-8").split("\n")
    current_branch = [b for b in branches if b.startswith("*")][0].split(" ")[1]
    assert current_branch == 'runs_log'

    subprocess.check_output(['git', 'add', '--all'])
    subprocess.check_output(['git', 'commit', '-am', 'update'])

    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
