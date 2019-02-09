import torch

from utils.util import folder_to_checkpoint


def resume_checkpoint(resume_path, model, logger, optimizers=None, lr_schedulers=None):
    if not resume_path.endswith(".pth"):
        resume_path = folder_to_checkpoint(resume_path)

    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])

    assert (optimizers is None and lr_schedulers is None) \
           or (optimizers is not None and lr_schedulers is not None)
    if optimizers is not None and lr_schedulers is not None:
        for opti_name in checkpoint['optimizers']:
            optimizers[opti_name].load_state_dict(checkpoint['optimizers'][opti_name])
        for lr_sched_name in checkpoint['lr_schedulers']:
            lr_schedulers[lr_sched_name].load_state_dict(checkpoint['lr_schedulers'][lr_sched_name])

    logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, start_epoch))
    return start_epoch


