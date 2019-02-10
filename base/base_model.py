import torch.nn as nn
import numpy as np

from utils.logger_config import logger

# @see https://stackoverflow.com/questions/3154460/python-human-readable-large-numbers/3155023
import math

millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.1f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def summary(self):
        """
        Model summary
        """

        logger.info('Trainable parameters: {}'.format(self.trainable_parameters()))
        logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        _trainable_parameters = self.trainable_parameters()
        return super(BaseModel, self).__str__() + '\nTrainable parameters: ~{} ({})'.format(
            millify(_trainable_parameters), _trainable_parameters)
