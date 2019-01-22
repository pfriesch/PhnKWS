import os
import re
import subprocess
import collections
import threading
import time

from utils.logger_config import Logger
from utils.tensorboard_logger import WriterTensorboardX


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
    # todo make sure exp folder is in gitignore
    branches = subprocess.check_output(['git', 'branch']).decode("utf-8").split("\n")
    current_branch = [b for b in branches if b.startswith("*")][0].split(" ")[1]
    assert current_branch == 'runs_log'

    subprocess.check_output(['git', 'add', '--all'])
    _diff = subprocess.check_output(['git', 'diff', '--exit-code'])
    if len(_diff) > 0:
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
                logger.exception("Problem while executing repetitive task. traceback: {}".format(e))


class Timer:

    def __init__(self, name, loggers, global_step=None):
        super().__init__()
        self.name = name
        self.loggers = loggers
        self.global_step = global_step

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        for logger in self.loggers:
            if isinstance(logger, WriterTensorboardX):
                logger.add_scalar(self.name, self.interval, global_step=self.global_step)
            elif isinstance(logger, Logger):
                logger.debug("{} took {:.5f}s".format(self.name, self.interval))

