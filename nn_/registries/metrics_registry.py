from nn_.metrics.metrics import LabCDAccuracy, LabMonoAccuracy, LabCDError, LabMonoError

from nn_.metrics.ctc_metrics import PhnErrorRate


def metrics_init(config):
    metrics = {}
    for metric in config['arch']['metrics']:
        if metric == 'acc_lab_cd':
            metrics[metric] = LabCDAccuracy()
        elif metric == 'acc_lab_mono':
            metrics[metric] = LabMonoAccuracy()
        elif metric == 'err_lab_mono':
            metrics[metric] = LabMonoError()
        elif metric == 'err_lab_cd':
            metrics[metric] = LabCDError()
        elif metric == 'phone_error_rate':
            metrics[metric] = PhnErrorRate(
                config['dataset']['dataset_definition']['data_info']['labels']['lab_phn']['num_lab'])
        else:
            raise ValueError("Can't find the metric {}".format(metric))
    return metrics
