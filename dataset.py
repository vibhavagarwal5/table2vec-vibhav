import argparse
import os
import time

import numpy as np
import torch
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils
from utils import Config, loadpkl, savepkl


class T2VDataset(Dataset):
    def __init__(self, X, y, vocab, device, config):
        self.config = config
        self.table_prep_params = config['table_prep_params']
        self.vocab = vocab

        # X = X.tolist()
        # for i in range(len(X)):
        #     X[i] = self.pad_table(X[i], '<PAD>')
        # X = self.table_words2index(X)
        # print(np.array(X).shape)

        self.X = torch.tensor(X, device=device)
        self.y = torch.tensor(y.tolist(), device=device, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def pad_table(table_prep_params, table, val):
        rows, cols = np.array(table).shape[:2]
        cols2fill = table_prep_params['MAX_COL_LEN'] - cols
        rows2fill = table_prep_params['MAX_ROW_LEN'] - rows

        for r in table:
            for cell in r:
                if len(cell) == 0:
                    cell.append(val)

        full_t = np.full(
            (
                table_prep_params['MAX_ROW_LEN'],
                table_prep_params['MAX_COL_LEN'],
                1), val).tolist()

        for i in range(int(rows2fill / 2), int(rows2fill / 2) + rows):
            full_t[i][int(cols2fill / 2):int(cols2fill / 2) +
                      cols] = table[i - int(rows2fill / 2)]
        return full_t

    @staticmethod
    def table_words2index(vocab, tables):
        w2i = {w: i for i, w in enumerate(vocab)}
        for t in tables:
            for r in t:
                for c in r:
                    for i, w in enumerate(c):
                        try:
                            c[i] = w2i[w]
                        except:
                            c[i] = w2i['<UNK>']
        return tables


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pad_data_prep", help="path for the scores", action='store_true')
    parser.add_argument("--path",
                        help="path for the datafiles")
    parser.add_argument("--xp_file",
                        help="Positive tables file")
    parser.add_argument("--vocab",
                        help="Vocab file")
    parser.add_argument("--max_col_len",
                        help="Max Column Length", type=int)
    parser.add_argument("--max_row_len",
                        help="Max Row Length", type=int)
    args = parser.parse_args()
    print(args)

    if args.pad_data_prep and \
            args.path and \
            args.xp_file and \
            args.vocab and \
            args.max_row_len and \
            args.max_col_len:

        Xp_path = args.path
        X = loadpkl(os.path.join(Xp_path, args.xp_file))
        vocab = loadpkl(os.path.join(Xp_path, args.vocab))
        table_prep_params = {
            "MAX_COL_LEN": args.max_col_len,
            "MAX_ROW_LEN": args.max_row_len
        }
        print(X.shape, len(vocab))

        X = X.tolist()
        print('Before padding')
        print(X[12])
        for i in range(len(X)):
            X[i] = T2VDataset.pad_table(table_prep_params, X[i], '<PAD>')
        print('After padding')
        print(X[12])

        X = T2VDataset.table_words2index(vocab, X)
        X = np.array(X)
        print('After w2i change')
        print(X[12])
        print(X.shape)

        temp = args.xp_file.split('.')
        temp[0] = temp[0] + '_pad_1'
        # savepkl(os.path.join(Xp_path, '.'.join(temp)), X)
    else:
        config = Config()
        device = torch.device(
            f"cuda:{1}" if torch.cuda.is_available() else 'cpu')
        dataset = T2VDataset(X, vocab, device, config)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        X_, y_ = next(iter(dataloader))
        print(X_.shape, y_.shape)

    print(time.time() - start)
