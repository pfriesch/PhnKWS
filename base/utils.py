import os
import random
import shutil
from glob import glob

import torch
import numpy as np

from utils.logger_config import logger
from utils.tensorboard_logger import WriterTensorboardX
from utils.util import folder_to_checkpoint
from utils.utils import run_shell


def set_seed(seed):
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # TODO add saving and loading of random state for reproducable research


def get_rng_state():
    torch_rng_state = torch.get_rng_state()
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    return torch_rng_state, python_rng_state, numpy_rng_state


def set_rng_state(torch_rng_state, python_rng_state, numpy_rng_state):
    torch.set_rng_state(torch_rng_state)
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)


def resume_checkpoint(resume_path, model, logger, optimizers=None, lr_schedulers=None,
                      seq_len_scheduler=None):
    if not resume_path.endswith(".pth"):
        resume_path = folder_to_checkpoint(resume_path)

    logger.info(f"Loading checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'dataset_sampler_state' not in checkpoint:
        checkpoint['dataset_sampler_state'] = None

    if checkpoint['dataset_sampler_state'] is None:
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    model.load_state_dict(checkpoint['state_dict'])

    assert (optimizers is None and lr_schedulers is None) \
           or (optimizers is not None and lr_schedulers is not None)
    if optimizers is not None and lr_schedulers is not None:
        for opti_name in checkpoint['optimizers']:
            optimizers[opti_name].load_state_dict(checkpoint['optimizers'][opti_name])
        for lr_sched_name in checkpoint['lr_schedulers']:
            lr_schedulers[lr_sched_name].load_state_dict(checkpoint['lr_schedulers'][lr_sched_name])

    logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, start_epoch))
    # TODO check checkpoint['dataset_sampler_state'] is none
    return start_epoch, global_step, checkpoint['dataset_sampler_state']


def save_checkpoint(epoch, global_step, model, optimizers, lr_schedulers, seq_len_scheduler, config,
                    checkpoint_dir,  # monitor_best=None,
                    dataset_sampler_state=None, save_best=None):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    assert dataset_sampler_state != save_best, "save_best is only done at the end of an epoch"

    # TODO figure out why shutil.disk_usage gives different result to df

    # available_disk_space_in_gb = shutil.disk_usage(checkpoint_dir).free * 1e-9
    available_disk_space_in_gb = run_shell(f"df -h {checkpoint_dir}")
    available_disk_space_in_gb = int(available_disk_space_in_gb.split("\n")[1].split(" ")[13][:-1])

    assert available_disk_space_in_gb > 5, \
        f"available_disk_space_in_gb of {available_disk_space_in_gb} is lower than 5GB" \
        + f"Aborting to try to save in order to not corrupt the model files"

    torch_rng_state, python_rng_state, numpy_rng_state = get_rng_state()

    state = {
        'epoch': epoch,
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizers': {opti_name: optimizers[opti_name].state_dict() for opti_name in optimizers},
        'lr_schedulers': {lr_sched_name: lr_schedulers[lr_sched_name].state_dict()
                          for lr_sched_name in lr_schedulers},
        'seq_len_scheduler': seq_len_scheduler,
        'dataset_sampler_state': dataset_sampler_state,
        # 'monitor_best': monitor_best,
        'config': config,
        'torch_rng_state': torch_rng_state,
        'python_rng_state': python_rng_state,
        'numpy_rng_state': numpy_rng_state,
    }
    if dataset_sampler_state is not None:
        # Intermediate save during training epoch
        all_previous_checkpoints = glob(os.path.join(checkpoint_dir, 'checkpoint_e*_gs*.pth'))
        checkpoint_name = f'checkpoint_e{epoch}_gs{global_step}.pth'

        filename = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(state, filename)
        logger.info(f"Saved checkpoint: {filename}")

        for old_checkpoint in all_previous_checkpoints:
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint} ")

    else:
        checkpoint_name = f'checkpoint_e{epoch}.pth'

        filename = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(state, filename)
        logger.info(f"Saved checkpoint: {filename}")

        if epoch >= 3:
            filename_prev = os.path.join(checkpoint_dir, f'checkpoint_e{epoch - 3}.pth')
            if os.path.exists(filename_prev):
                os.remove(filename_prev)
                logger.info(f"Removed old checkpoint: {filename_prev} ")

        if save_best is not None and save_best:
            checkpoint_name = f'checkpoint_best.pth'

            best_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(state, best_path)
            logger.info(f"Saved current best: {checkpoint_name}")

    # available_disk_space_in_gb = shutil.disk_usage(checkpoint_dir).free * 1e-9
    available_disk_space_in_gb = run_shell(f"df -h {checkpoint_dir}")
    available_disk_space_in_gb = int(available_disk_space_in_gb.split("\n")[1].split(" ")[13][:-1])
    assert available_disk_space_in_gb > 5, \
        f"available_disk_space_in_gb of {available_disk_space_in_gb} is lower than 5GB" \
        + f"Aborting since next checkpoint save probably fails because of too little space -> no wasted training compute"
