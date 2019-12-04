import logging
import numpy as np
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.nn.init import xavier_normal_, xavier_uniform_

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
        xavier_normal_(self.embeddings.weight.data)

        self.conv1 = nn.Conv2d(embedding_dim, 128, (3, 2), 1, 0)
        self.mp1 = nn.MaxPool2d((2, 2), 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128*24*4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        emb = torch.from_numpy(np.average(emb.cpu().data.numpy(), axis=3))
        emb = emb.to(self.device)
        emb = emb.transpose(1, 3).contiguous().transpose(2, 3).contiguous()

        x = self.conv1(emb)
        x = self.bn1(x)
        x = self.mp1(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.drop2(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        pred = torch.sigmoid(x)
        return pred
