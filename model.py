import logging
import numpy as np
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from preprocess import loadpkl, savepkl, print_table

logger = logging.getLogger("app")

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        logger.info(x.shape)
        return x


class Table2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, device):
        super(Table2Vec, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3),
            #             PrintLayer(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #             PrintLayer(),
            nn.Conv2d(128, 64, kernel_size=3),
            #             PrintLayer(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #             PrintLayer(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64*11*1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        emb = torch.from_numpy(np.average(emb.cpu().data.numpy(), axis=3))
        emb = emb.to(self.device)
        emb = emb.transpose(1, 3).contiguous().transpose(2, 3).contiguous()
#         emb = emb.view(emb.shape[0],emb.shape[-1],emb.shape[1],emb.shape[2])

        x = self.cnn_layers(emb)
        x = x.view(x.shape[0], -1)
        x = self.linear_layers(x)
        return x
