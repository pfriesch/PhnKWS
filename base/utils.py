import os

import torch

from data.data_util import load_counts
from utils.util import folder_to_checkpoint


def resume_checkpoint(resume_path, model, logger, optimizers=None, lr_schedulers=None):
    if not resume_path.endswith(".pth"):
        resume_path = folder_to_checkpoint(resume_path)

    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    if 'global_step' not in checkpoint:
        # backwards compability
        checkpoint['global_step'] = -1
    if 'decoding_norm_data' not in checkpoint:
        # backwards compability
        # load_counts(config['test'][output_label]['normalize_with_counts_from_file'])

        decoding_norm_data = {}
        if 'test' in checkpoint['config']:
            for test_label in checkpoint['config']['test']:
                if 'normalize_with_counts_from_file' in checkpoint['config']['test'][test_label]:
                    if not os.path.exists(checkpoint['config']['test'][test_label]['normalize_with_counts_from_file']):
                        print("File does not exists: " + os.path.abspath(
                            checkpoint['config']['test'][test_label]['normalize_with_counts_from_file']))
                        # if "/mnt/data/pytorch-kaldi/exp" in checkpoint['config']['test'][test_label][
                        #     'normalize_with_counts_from_file']:
                        checkpoint['config']['test'][test_label]['normalize_with_counts_from_file'] = \
                            os.path.abspath(
                                checkpoint['config']['test'][test_label]['normalize_with_counts_from_file'])
                        checkpoint['config']['test'][test_label]['normalize_with_counts_from_file'] = \
                            checkpoint['config']['test'][test_label]['normalize_with_counts_from_file'] \
                                .replace("/mnt/data/pytorch-kaldi/exp",
                                         "/mnt/data/pytorch-kaldi/exp_TIMIT_TDNN_FBANK")
                        print(checkpoint['config']['test'][test_label]['normalize_with_counts_from_file'])
                        print(
                            os.path.exists(checkpoint['config']['test'][test_label]['normalize_with_counts_from_file']))

                    decoding_norm_data[test_label] = {}
                    decoding_norm_data[test_label]["normalize_with_counts"] = \
                        load_counts(checkpoint['config']['test'][test_label]['normalize_with_counts_from_file'])
        checkpoint['decoding_norm_data'] = decoding_norm_data

    decoding_norm_data = checkpoint['decoding_norm_data']
    start_global_step = checkpoint['global_step']
    model.load_state_dict(checkpoint['state_dict'])

    assert (optimizers is None and lr_schedulers is None) \
           or (optimizers is not None and lr_schedulers is not None)
    if optimizers is not None and lr_schedulers is not None:
        for opti_name in checkpoint['optimizers']:
            optimizers[opti_name].load_state_dict(checkpoint['optimizers'][opti_name])
        for lr_sched_name in checkpoint['lr_schedulers']:
            lr_schedulers[lr_sched_name].load_state_dict(checkpoint['lr_schedulers'][lr_sched_name])

    logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, start_epoch))
    return start_epoch, start_global_step, decoding_norm_data
