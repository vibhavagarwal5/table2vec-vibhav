import logging
import torch.nn as nn

logger = logging.getLogger("app")


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        logger.info(x.shape)
        return x
