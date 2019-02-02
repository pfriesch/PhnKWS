import torch.nn as nn

from nn_.utils.CNN_utils import LayerStats


class BaseLayer(nn.Module):

    def __init__(self):
        super(BaseLayer, self).__init__()

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def getself_layer_stats(self) -> [LayerStats]:
        """
        list of LayerStats
        """
        raise NotImplementedError
