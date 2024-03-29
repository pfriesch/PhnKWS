import os
import re
import subprocess
import collections
import threading
import time
from glob import glob

from utils.logger_config import Logger
from utils.tensorboard_logger import WriterTensorboardX


def recursive_update(_dict, _update_dict):
    if isinstance(_dict, str):
        return _update_dict
    for k, v in _update_dict.items():
        if isinstance(v, collections.Mapping):
            _dict[k] = recursive_update(_dict.get(k, {}), v)
        else:
            _dict[k] = v

    return _dict


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def folder_to_checkpoint(checkpoint_folder_path):
#     if not checkpoint_folder_path.endswith("checkpoints"):
#         checkpoint_folder_path = os.path.join(checkpoint_folder_path, 'checkpoints')
#     checkpoints = glob(os.path.join(checkpoint_folder_path, 'checkpoint_e*_gs*.pth'))
#     assert len(checkpoints) == 1, "expected only one checkpoint to be present, did something go wrong during saving?"
#
#     # checkpoints = [(re.findall("checkpoint-epoch(.*)\.pth", cp), cp) for cp in checkpoints]
#     # resume_path = max(checkpoints, key=lambda x: x[0])[1]
#     return os.path.join(checkpoint_folder_path, checkpoints[0])

def folder_to_checkpoint(checkpoint_folder_path):
    if not checkpoint_folder_path.endswith("checkpoints"):
        checkpoint_folder_path = os.path.join(checkpoint_folder_path, 'checkpoints')
    checkpoints = glob(os.path.join(checkpoint_folder_path, 'checkpoint*.pth'))

    checkpoints_str = [f"[{chkpnt_id}]: {chkpnt}\n" for chkpnt_id, chkpnt in enumerate(checkpoints)]
    print(f"Found these checkpoints: \n{''.join(checkpoints_str)}")
    _input = input("Enter checkpoint ID: ")
    chosen_checkpoint = int(_input)
    checkpoint = checkpoints[chosen_checkpoint]

    # assert len(checkpoints) == 1, "expected only one checkpoint to be present, did something go wrong during saving?"

    # checkpoints = [(re.findall("checkpoint-epoch(.*)\.pth", cp), cp) for cp in checkpoints]
    # resume_path = max(checkpoints, key=lambda x: x[0])[1]
    return os.path.join(checkpoint_folder_path, checkpoint)


def code_versioning():
    # todo make sure exp folder is in gitignore
    branches = subprocess.check_output(['git', 'branch']).decode("utf-8").split("\n")
    current_branch = [b for b in branches if b.startswith("*")][0].split(" ")[1]
    assert current_branch == 'runs_log'
    assert not os.path.exists(".git/index.lock")

    subprocess.check_output(['git', 'add', '--all'])
    _diff = subprocess.check_output(['git', 'status'])
    if _diff != b'On branch runs_log\nnothing to commit, working tree clean\n':
        _ret_val = subprocess.check_output(['git', 'commit', '-am', 'update'])

    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()


def every(delay, task, logger, stop_switch: threading.Event):
    while True:
        stop = stop_switch.wait(delay)
        if stop:
            return 0
        else:
            try:
                task()
            except Exception as e:
                # traceback.print_exc()
                # in production code you might want to have this instead of course:
                logger.exception(f"Problem while executing repetitive task. traceback: {e}")


class Timer:

    def __init__(self, name, loggers, global_step=None):
        super().__init__()
        self.name = name
        self.loggers = loggers
        self.global_step = global_step

    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start
        for logger in self.loggers:
            if isinstance(logger, WriterTensorboardX):
                logger.add_scalar(self.name, self.interval, global_step=self.global_step, mode="info")
            elif isinstance(logger, Logger):
                logger.debug(f"{self.name} took {self.interval:.2f}s")
