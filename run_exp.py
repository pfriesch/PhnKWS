import os
import argparse
import datetime

from nn_ import model_init, optimizer_init, lr_scheduler_init, metrics_init, loss_init
from utils.logger_config import logger
from utils.nvidia_smi import nvidia_smi_enabled
from utils.util import code_versioning
from utils.utils import set_seed
from trainer import Trainer
from utils.utils import check_environment, read_json
from utils.utils import get_dataset_metadata


def check_config(config):
    # TODO impl schema or sth
    pass


def setup_run(config):
    set_seed(config['exp']['seed'])
    config, decoding_norm_data = get_dataset_metadata(config)

    model = model_init(config)

    optimizers = optimizer_init(config, model)

    lr_schedulers = lr_scheduler_init(config, optimizers)

    logger.debug(model)
    metrics = metrics_init(config)

    loss = loss_init(config)

    return model, loss, metrics, optimizers, config, lr_schedulers, decoding_norm_data


def main(config_path, resume_path, overfit_small_batch):
    config = read_json(config_path)
    check_config(config)

    if overfit_small_batch:
        config['exp']['num_workers'] = 0

    # if resume_path:
    # TODO
    #     resume_config = torch.load(folder_to_checkpoint(args.resume), map_location='cpu')['config']
    #     # also the results won't be the same give the different random seeds with different number of draws
    #     del config['exp']['name']
    #     recursive_update(resume_config, config)
    #
    #     print("".join(["="] * 80))
    #     print("Resume with these changes in the config:")
    #     print("".join(["-"] * 80))
    #     print(jsondiff.diff(config, resume_config, dump=True, dumper=jsondiff.JsonDumper(indent=1)))
    #     print("".join(["="] * 80))
    #
    #     config = resume_config
    #     # start_time = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    #     # config['exp']['name'] = config['exp']['name'] + "r-" + start_time
    # else:
    start_time = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    config['exp']['name'] = config['exp']['name'] + start_time

    set_seed(config['exp']['seed'])

    config['exp']['save_dir'] = os.path.abspath(config['exp']['save_dir'])

    # Output folder creation
    out_folder = os.path.join(config['exp']['save_dir'], config['exp']['name'])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder + '/exp_files')

    logger.configure_logger(out_folder)

    check_environment()

    if nvidia_smi_enabled:  # TODO chage criteria or the whole thing
        git_commit = code_versioning()
        if 'versioning' not in config:
            config['versioning'] = {}
        config['versioning']['git_commit'] = git_commit

    logger.info("Experiment name : {}".format(out_folder))
    logger.info("tensorboard : tensorboard --logdir {}".format(os.path.abspath(out_folder)))

    model, loss, metrics, optimizers, config, lr_schedulers, decoding_norm_data = setup_run(config)

    trainer = Trainer(model, loss, metrics, optimizers, lr_schedulers, decoding_norm_data,
                      resume_path, config,
                      do_validation=True,
                      overfit_small_batch=overfit_small_batch)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-o', '--overfit', action='store_true',
                        help='overfit_small_batch / debug mode')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.config, args.resume, args.overfit)
