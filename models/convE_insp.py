import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_, xavier_uniform_
from torchsummary import summary

from utils import flatten_1_deg, loadpkl, savepkl


class Table2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, device):
        super(Table2Vec, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = True
        xavier_normal_(self.embeddings.weight.data)

        self.conv1 = nn.Conv2d(embedding_dim, 128, (3, 2), 1, 0)
        self.mp1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 4 * 1, 64)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        emb = emb.mean(3)
        emb = emb.permute(0, 3, 1, 2)

        x = self.conv1(emb)
        x = self.bn1(x)
        x = self.mp1(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.drop2(x)
        # x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        pred = torch.sigmoid(x)
        return pred
