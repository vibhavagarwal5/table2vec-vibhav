import numpy as np
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.nn.init import xavier_normal_, xavier_uniform_

from preprocess import loadpkl, savepkl, print_table


class Table2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, device):
        super(Table2Vec, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = True
        xavier_normal_(self.embeddings.weight.data)

        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(100*50*10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        emb = emb.mean(3)
        emb = emb.permute(0, 3, 1, 2)
        x = emb.reshape(emb.shape[0],-1)
        # x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        pred = torch.sigmoid(x)
        return pred
