import argparse
from torch.multiprocessing import Pool
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils
from utils import Config, loadpkl, savepkl
from preprocess import cell_overflow_cap, print_table


class T2VDataset(Dataset):
    def __init__(self, X, y, vocab, device, config):
        self.config = config
        self.table_prep_params = config['table_prep_params']
        self.vocab = vocab

        # X = cell_overflow_cap(X)
        # p = Pool(processes=20)
        # X = p.map(self.pad_table, X)
        # p.close()
        # p.join()
        # X = self.table_words2index(X)

        self.X = torch.tensor(X, device=device)
        self.y = torch.tensor(y.tolist(), device=device, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # NOTE: MAKE THIS FAST...
    def pad_table(self, table):
        rows = len(table)
        for row in table:
            for cell in row:
                for i in range(0, self.table_prep_params['LENGTH_PER_CELL']-len(cell)):
                    cell.append('<PAD>')
            for j in range(0, self.table_prep_params['MAX_COL_LEN']-len(row)):
                row.append(['<PAD>']*self.table_prep_params['LENGTH_PER_CELL'])
        for i in range(0, self.table_prep_params['MAX_ROW_LEN']-rows):
            table.append([['<PAD>']*self.table_prep_params['LENGTH_PER_CELL']]
                         * self.table_prep_params['MAX_COL_LEN'])
        return table

    def table_words2index(self, tables):
        w2i = {w: i for i, w in enumerate(self.vocab)}
        for i, t in enumerate(tables):
            tables[i] = np.vectorize(
                lambda y: w2i[y])(np.array(t)).tolist()
        return tables


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pad_data_prep", help="path for the scores", action='store_true')
    parser.add_argument(
        "--data_type", help="+ve/-ve data type to prepare")
    args = parser.parse_args()

    config = Config()

    if args.data_type == '+':
        X = loadpkl(config['input_files']['Xp_path'])
        y = loadpkl(config['input_files']['yp_path'])
    elif args.data_type == '-':
        X = loadpkl(config['input_files']['Xn_path'])
        y = loadpkl(config['input_files']['yn_path'])
    vocab = loadpkl(config['input_files']['vocab_path'])
    table_prep_params = config['table_prep_params']
    print(X.shape, y.shape, len(vocab))

    if args.pad_data_prep:
        def pad_table(table):
            rows = len(table)
            for row in table:
                for cell in row:
                    for i in range(0, table_prep_params['LENGTH_PER_CELL']-len(cell)):
                        cell.append('<PAD>')
                for j in range(0, table_prep_params['MAX_COL_LEN']-len(row)):
                    row.append(['<PAD>']*table_prep_params['LENGTH_PER_CELL'])
            for i in range(0, table_prep_params['MAX_ROW_LEN']-rows):
                table.append([['<PAD>']*table_prep_params['LENGTH_PER_CELL']]
                             * table_prep_params['MAX_COL_LEN'])
            return table

        def table_words2index(tables):
            w2i = {w: i for i, w in enumerate(vocab)}
            for i, t in enumerate(tables):
                tables[i] = np.vectorize(
                    lambda y: w2i[y])(np.array(t)).tolist()
            return tables

        X = cell_overflow_cap(X)
        p = Pool(processes=40)
        X = p.map(pad_table, X)
        p.close()
        p.join()
        X = table_words2index(X)
        X = np.array(X)
        print(X.shape)
        if args.data_type == '+':
            savepkl('./data/xp_2D_10-50_pad.pkl', X)
        elif args.data_type == '-':
            savepkl('./data/xn_2D_10-50_pad.pkl', X)
    else:
        device = torch.device(
            f"cuda:{1}" if torch.cuda.is_available() else 'cpu')
        dataset = T2VDataset(X, y, vocab, device, config)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        X_, y_ = next(iter(dataloader))
        print(X_.shape, y_.shape)

    print(time.time()-start)
