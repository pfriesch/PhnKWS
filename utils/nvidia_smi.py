nvidia_smi_enabled = False
try:
    import nvidia_smi

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # gpu0
    # TODO handle mutiple gpus
    nvidia_smi_enabled = True


except Exception as e:
    import warnings

    if isinstance(e, ImportError):
        warnings.warn('GPU usage loggin disabled. Install nvidia-ml-py or nvidia-ml-py3 to enable it.')
    else:
        warnings.warn(str(e))


def get_gpu_usage():
    if nvidia_smi_enabled:
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        return res.gpu
    else:
        return -1


def get_gpu_memory_consumption():
    if nvidia_smi_enabled:
        res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return int(res.used / 10e5)
    else:
        return -1
