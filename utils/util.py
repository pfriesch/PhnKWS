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


def check_code_versioning():
    diff = subprocess.check_output(['git', 'diff'])
    assert len(diff) == 0
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])
