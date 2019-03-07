from utils.logger_config import logger


class IncreaseSeqLenAfterEpoch(object):

    def __init__(self, max_seq_length_train,
                 start_seq_len_train,
                 increase_seq_length_train,
                 multply_factor_seq_len_train, verbose=True):

        self.max_seq_length_train = max_seq_length_train
        # self.start_seq_len_train = start_seq_len_train
        self.increase_seq_length_train = increase_seq_length_train
        self.multply_factor_seq_len_train = multply_factor_seq_len_train
        self.verbose = verbose

        self.max_seq_length_train_curr = start_seq_len_train

    def step(self):
        if self.increase_seq_length_train:
            self.max_seq_length_train_curr *= self.multply_factor_seq_len_train
            if self.max_seq_length_train_curr > self.max_seq_length_train:
                self.max_seq_length_train_curr = self.max_seq_length_train
                if self.verbose:
                    logger.info(f"max_seq_length_train_curr set to {self.max_seq_length_train_curr}")

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict, tensorboard_logger=None):
        self.__dict__.update(state_dict)


def seq_len_scheduler_init(config):
    seq_len_scheduler = IncreaseSeqLenAfterEpoch(config['training']['batching']['max_seq_length_train'],
                                                 config['training']['batching']['start_seq_len_train'],
                                                 config['training']['batching']['increase_seq_length_train'],
                                                 config['training']['batching']['multply_factor_seq_len_train'],
                                                 verbose=True)

    return seq_len_scheduler
