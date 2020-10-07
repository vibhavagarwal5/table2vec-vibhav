import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning import _logger as log
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics import Accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from utils import loadpkl


class NDCGDataset(Dataset):
    def __init__(self, df):
        x_f = ['row', 'col', 'nul', 'in_link', 'out_link', 'pgcount', 'tImp', 'tPF', 'leftColhits', 'SecColhits', 'bodyhits', 'PMI', 'qInPgTitle', 'qInTableTitle', 'yRank', 'csr_score', 'idf1', 'idf2',
               'idf3', 'idf4', 'idf5', 'idf6', 'max', 'sum', 'avg', 'sim', 'emax', 'esum', 'eavg', 'esim', 'cmax', 'csum', 'cavg', 'csim', 'remax', 'resum', 'reavg', 'resim', 'query_l'] + ['cos_sim']
        y_f = ['rel']
        self.X = df[x_f].to_numpy()
        self.y = df[y_f].to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).float()
        y = torch.tensor(self.y[idx])
        return x, y


class NDCG_MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(40, 20)
        self.dp1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(20, 40)
        self.out_proj = nn.Linear(40, 3)

        self.acc = Accuracy()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def setup(self, stage):
        df = loadpkl(
            '/home/vibhav_student/2Dcnn/data/w_all_data/baseline_f_tq-tkn-pad_qfix_shuffle_cossim.csv')
        dataset = NDCGDataset(df)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(
            dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, num_workers=4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.acc(y_hat.argmax(dim=1), y)
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.acc(y_hat.argmax(dim=1), y)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


if __name__ == "__main__":
    seed_everything(0)

    model = NDCG_MLP()

    trainer = Trainer(gpus=0, max_epochs=10)
    trainer.fit(model)
