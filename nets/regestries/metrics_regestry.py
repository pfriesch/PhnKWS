from nets.metrics.metrics import LabCDAccuracy, LabMonoAccuracy


def metrics_init(config):
    metrics = {}
    for metric in config['arch']['metrics']:
        if metric == 'acc_lab_cd':
            metrics[metric] = LabCDAccuracy()
        elif metric == 'acc_lab_mono':
            metrics[metric] = LabMonoAccuracy()
        else:
            raise ValueError("Can't find the metric {}".format(metric))
    return metrics
