from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import utils
from preprocess import loadpkl


class T2VDataset(Dataset):
    def __init__(self, X, y, vocab, device, config):
        self.config = config
        self.table_prep_params = config['table_prep_params']
        self.vocab = vocab

        p = Pool(processes=20)
        X = p.map(self.pad_table, X)
        p.close()
        p.join()
        X = self.table_words2index(X)

        self.X = torch.tensor(X, device=device)
        self.y = torch.tensor(y.tolist(), device=device, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def pad_table(self, table):
        rows = len(table)
        for row in table:
            for cell in row:
                for i in range(self.table_prep_params['LENGTH_PER_CELL']-len(cell)):
                    cell.append('<PAD>')
            for j in range(self.table_prep_params['MAX_COL_LEN']-len(row)):
                row.append(['<PAD>']*self.table_prep_params
                           ['LENGTH_PER_CELL'])
        for i in range(self.table_prep_params['MAX_ROW_LEN']-rows):
            table.append([['<PAD>']*self.table_prep_params['LENGTH_PER_CELL']]
                         * self.table_prep_params['MAX_COL_LEN'])
        return table

    def table_words2index(self, tables):
        w2i = {w: i for i, w in enumerate(self.vocab)}

        for i, t in enumerate(tables):
            t = np.array(t)
            vf = np.vectorize(lambda x: w2i[x])
            tables[i] = vf(t)
        return tables
        # for i, row in enumerate(t):
        #     for j, cell in enumerate(row):
        #         for k, item in enumerate(cell):
        #             t[i][j][k] = w2i[item]


if __name__ == '__main__':
    X = loadpkl('./data/X_2D_10-50_tkn.pkl')
    y = loadpkl('./data/y_2D_10-50.pkl')
    vocab = loadpkl('./data/vocab_2D_10-50.pkl')
    print(X.shape, y.shape, len(vocab))

    device = torch.device(
        f"cuda:{1}" if torch.cuda.is_available() else 'cpu')
    dataset = T2VDataset(X, y, vocab, device, utils.load_config_args())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    X_, y_ = next(iter(dataloader))
    print(X_.shape, y_.shape)

