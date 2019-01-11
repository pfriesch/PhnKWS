import argparse
import datetime
import json
import os

import jsondiff
import torch

from nets import model_init, optimizer_init, lr_scheduler_init, metrics_init, loss_init
from utils.logger_config import logger
from utils.util import code_versioning, folder_to_checkpoint, recursive_update
from utils.utils import set_seed, get_posterior_norm_data
from trainer import Trainer
from utils.utils import check_environment, read_json

check_environment()


def setup_run(config):
    set_seed(config['exp']['seed'])

    config, N_out_lab = get_posterior_norm_data(config)

    model = model_init(config['arch']['name'], fea_index_length=config['arch']['args']['fea_index_length'],
                       lab_cd_num=N_out_lab['lab_cd'])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer_init(config, trainable_params)

    lr_scheduler = lr_scheduler_init(config, optimizer)

    logger.debug(model)
    metrics = metrics_init(config)

    loss = loss_init(config)

    return model, loss, metrics, optimizer, config, lr_scheduler


def main(config_path, resume_path, debug):
    config = read_json(config_path)
    if not debug:
        git_commit = code_versioning()
        if 'versioning' not in config:
            config['versioning'] = {}
        config['versioning']['git_commit'] = git_commit

    if resume_path:
        resume_config = torch.load(folder_to_checkpoint(args.resume))['config']
        # also the results won't be the same give the different random seeds with different number of draws
        del config['exp']['name']
        recursive_update(resume_config, config)

        result = jsondiff.diff(config, resume_config)
        print("".join(["="] * 80))
        print("Resume with these changes in the config:")
        print("".join(["-"] * 80))
        print(json.dumps(result, indent=1))
        print("".join(["="] * 80))

        config = resume_config
        start_time = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
        config['exp']['name'] = config['exp']['name'] + "r-" + start_time
    else:
        start_time = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
        config['exp']['name'] = config['exp']['name'] + start_time

    set_seed(config['exp']['seed'])

    # Output folder creation
    out_folder = os.path.join(config['exp']['save_dir'], config['exp']['name'])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder + '/exp_files')

    logger.configure_logger(out_folder)

    model, loss, metrics, optimizer, config, lr_scheduler = setup_run(config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume_path=resume_path,
                      config=config,
                      do_validation=True,
                      lr_scheduler=lr_scheduler,
                      debug=debug)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Run in debug mode with few samples')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.config, args.resume, debug=args.debug)
