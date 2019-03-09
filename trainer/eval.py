import logging
import os
from glob import glob
from multiprocessing import Manager
from multiprocessing.pool import Pool

from tqdm import tqdm

from base.base_trainer import to_device, detach_cpu
from base.base_trainer import KaldiOutputWriter, metrics_accumulator
from data.dataset_regestry import get_dataset
from data.datasets import DatasetType
from data.kaldi_data_loader import KaldiDataLoader
from kaldi_decoding_scripts.decode_dnn import decode as decode_ce, best_wer
from kaldi_decoding_scripts.ctc_decoding.decode_dnn import decode_ctc
from utils.logger_config import logger


def evaluate(model, metrics, device, out_folder, exp_name, max_label_length, epoch, dataset_type, data_cache_root,
             test_with,
             all_feats_dict, features_use, all_labs_dict, labels_use, phoneme_dict, decoding_info, lab_graph_dir=None,
             tensorboard_logger=None):
    model.eval()
    batch_size = 1
    max_seq_length = -1

    accumulated_test_metrics = {metric: 0 for metric in metrics}

    test_data = test_with
    dataset = get_dataset(dataset_type,
                          data_cache_root,
                          f"{test_data}_{exp_name}",
                          {feat: all_feats_dict[feat] for feat in features_use},
                          {lab: all_labs_dict[lab] for lab in labels_use},
                          max_seq_length,
                          model.context_left,
                          model.context_right,
                          normalize_features=True,
                          phoneme_dict=phoneme_dict,
                          max_seq_len=max_seq_length,
                          max_label_length=max_label_length)

    dataloader = KaldiDataLoader(dataset,
                                 batch_size,
                                 use_gpu=False,
                                 batch_ordering=model.batch_ordering)

    assert len(dataset) >= batch_size, \
        f"Length of test dataset {len(dataset)} too small " \
        + f"for batch_size of {batch_size}"

    n_steps_this_epoch = 0
    warned_size = False

    with Pool(os.cpu_count()) as pool:
        multip_process = Manager()
        metrics_q = multip_process.Queue(maxsize=os.cpu_count())
        # accumulated_test_metrics_future_list = pool.apply_async(metrics_accumulator, (metrics_q, metrics))
        accumulated_test_metrics_future_list = [pool.apply_async(metrics_accumulator, (metrics_q, metrics))
                                                for _ in range(os.cpu_count())]
        with KaldiOutputWriter(out_folder, test_data, model.out_names, epoch) as writer:
            with tqdm(disable=not logger.isEnabledFor(logging.INFO), total=len(dataloader), position=0) as pbar:
                pbar.set_description('E e:{}    '.format(epoch))
                for batch_idx, (sample_names, inputs, targets) in enumerate(dataloader):
                    n_steps_this_epoch += 1

                    if batch_idx > 10:
                        break

                    inputs = to_device(device, inputs)
                    if "lab_phn" not in targets:
                        targets = to_device(device, targets)

                    output = model(inputs)

                    output = detach_cpu(output)
                    targets = detach_cpu(targets)

                    #### Logging ####
                    metrics_q.put((output, targets))

                    pbar.set_description('E e:{} '.format(epoch))
                    pbar.update()
                    #### /Logging ####

                    warned_label = False
                    for output_label in output:
                        if output_label in model.out_names:
                            # squeeze that batch
                            output[output_label] = output[output_label].squeeze(1)
                            # remove blank/padding 0th dim
                            # if config["arch"]["framewise_labels"] == "shuffled_frames":
                            out_save = output[output_label].data.cpu().numpy()
                            # else:
                            #     raise NotImplementedError("TODO make sure the right dimension is taken")
                            #     out_save = output[output_label][:, :-1].data.cpu().numpy()

                            if len(out_save.shape) == 3 and out_save.shape[0] == 1:
                                out_save = out_save.squeeze(0)

                            if dataset.state.dataset_type != DatasetType.SEQUENTIAL_APPENDED_CONTEXT \
                                    and dataset.state.dataset_type != DatasetType.SEQUENTIAL:
                                raise NotImplementedError("TODO rescaling with prior")

                            # if config['dataset']['dataset_definition']['decoding']['normalize_posteriors']:
                            #     # read the config file
                            #     counts = config['dataset']['dataset_definition'] \
                            #         ['data_info']['labels']['lab_phn']['lab_count']
                            #     if out_save.shape[-1] == len(counts) - 1:
                            #         if not warned_size:
                            #             logger.info(
                            #                 f"Counts length is {len(counts)} but output"
                            #                 + f" has size {out_save.shape[-1]}."
                            #                 + f" Assuming that counts is 1 indexed")
                            #             warned_size = True
                            #         counts = counts[1:]
                            #     # Normalize by output count
                            # #     if ctc:
                            # #         blank_scale = 1.0
                            # #         # TODO try different blank_scales 4.0 5.0 6.0 7.0
                            # #         counts[0] /= blank_scale
                            # #         # for i in range(1, 8):
                            # #         #     counts[i] /= noise_scale #TODO try noise_scale for SIL SPN etc I guess
                            # #
                            # #     prior = np.log(counts / np.sum(counts))
                            #
                            #     out_save = out_save - np.log(prior)

                            # shape == NC
                            assert len(out_save.shape) == 2
                            assert len(sample_names) == 1
                            writer.write_mat(output_label, out_save.squeeze(), sample_names[0])


                        else:
                            if not warned_label:
                                logger.debug("Skipping saving forward for decoding for key {}".format(output_label))
                                warned_label = True

            for _accumulated_test_metrics in accumulated_test_metrics_future_list:
                metrics_q.put(None)
            for _accumulated_test_metrics in accumulated_test_metrics_future_list:
                _accumulated_test_metrics = _accumulated_test_metrics.get()
                for metric, metric_value in _accumulated_test_metrics.items():
                    accumulated_test_metrics[metric] += metric_value

    # test_metrics = {metric: 0 for metric in metrics}
    # for metric in accumulated_test_metrics:
    #     for metric, metric_value in metric.items():
    #         test_metrics[metric] += metric_value

    test_metrics = {metric: accumulated_test_metrics[metric] / len(dataloader)
                    for metric in accumulated_test_metrics}
    if tensorboard_logger is not None:
        tensorboard_logger.set_step(epoch, 'eval')
        for metric, metric_value in test_metrics.items():
            tensorboard_logger.add_scalar(metric, test_metrics[metric] / len(dataloader))

    # decoding_results = []
    #### DECODING ####
    # for out_lab in model.out_names:
    out_lab = model.out_names[0]  # TODO query from model or sth

    # forward_data_lst = config['data_use']['test_with'] #TODO multiple forward sets
    # forward_data_lst = [config['dataset']['data_use']['test_with']]
    # forward_dec_outs = config['test'][out_lab]['require_decoding']

    # for data in forward_data_lst:
    logger.debug('Decoding {} output {}'.format(test_with, out_lab))

    if out_lab == 'out_cd':
        _label = 'lab_cd'
    elif out_lab == 'out_phn':
        _label = 'lab_phn'
    else:
        raise NotImplementedError(out_lab)

    lab_field = all_labs_dict[_label]

    out_folder = os.path.abspath(out_folder)
    out_dec_folder = '{}/decode_{}_{}'.format(out_folder, test_with, out_lab)

    # logits_test_clean_100_ep006_out_phn.ark
    files_dec_list = glob(f'{out_folder}/exp_files/logits_{test_with}_ep*_{out_lab}.ark')

    if lab_graph_dir is None:
        lab_graph_dir = os.path.abspath(lab_field['lab_graph'])
    if _label == 'lab_phn':
        decode_ctc(data=os.path.abspath(lab_field['lab_data_folder']),
                   graphdir=lab_graph_dir,
                   out_folder=out_dec_folder,
                   featstrings=files_dec_list)
    elif _label == 'lab_cd':
        decode_ce(**decoding_info,
                  alidir=os.path.abspath(lab_field['label_folder']),
                  data=os.path.abspath(lab_field['lab_data_folder']),
                  graphdir=lab_graph_dir,
                  out_folder=out_dec_folder,
                  featstrings=files_dec_list)
    else:
        raise ValueError(_label)

    decoding_results = best_wer(out_dec_folder, decoding_info['scoring_type'])
    logger.info(decoding_results)

    tensorboard_logger.add_text("WER results", str(decoding_results))

    # TODO plotting curves

    return {'test_metrics': test_metrics, "decoding_results": decoding_results}
