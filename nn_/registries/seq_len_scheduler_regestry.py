from utils.logger_config import logger


class _SeqLenScheduler(object):
    def __init__(self, last_epoch=-1):
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_seq_len(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch


class IncreaseSeqLenAfterEpoch(_SeqLenScheduler):

    def __init__(self, max_seq_length_train,
                 start_seq_len_train,
                 increase_seq_length_train,
                 multply_factor_seq_len_train, last_epoch=0, verbose=True):

        self.max_seq_length_train = max_seq_length_train
        self.start_seq_len_train = start_seq_len_train
        self.increase_seq_length_train = increase_seq_length_train
        self.multply_factor_seq_len_train = multply_factor_seq_len_train
        self.verbose = verbose

        super(IncreaseSeqLenAfterEpoch, self).__init__(last_epoch)

    def get_seq_len(self, epoch=None):

        if epoch is None:
            epoch = self.last_epoch

        max_seq_length_train_curr = self.start_seq_len_train
        if self.increase_seq_length_train:
            max_seq_length_train_curr = self.start_seq_len_train * (
                    self.multply_factor_seq_len_train ** epoch)
            if max_seq_length_train_curr > self.max_seq_length_train:
                max_seq_length_train_curr = self.max_seq_length_train
                if self.verbose:
                    logger.info(f"max_seq_length_train_curr set to {max_seq_length_train_curr}")
        return max_seq_length_train_curr


def seq_len_scheduler_init(config):
    seq_len_scheduler = IncreaseSeqLenAfterEpoch(config['training']['batching']['max_seq_length_train'],
                                                 config['training']['batching']['start_seq_len_train'],
                                                 config['training']['batching']['increase_seq_length_train'],
                                                 config['training']['batching']['multply_factor_seq_len_train'],
                                                 verbose=True)

    return seq_len_scheduler
