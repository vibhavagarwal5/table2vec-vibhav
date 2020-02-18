import numpy as np
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from preprocess import loadpkl, savepkl, print_table

class Table2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, device):
        super(Table2Vec, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = True
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64*11*1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        emb = emb.mean(3)
        emb = emb.permute(0, 3, 1, 2)

        x = self.cnn_layers(emb)
        x = x.view(x.shape[0], -1)
        x = self.linear_layers(x)
        return x
