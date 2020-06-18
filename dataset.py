import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import Config, loadpkl, savepkl


class NegSample():
    def __init__(self, all_tables, config):
        self.all_tables = all_tables
        self.config = config
        self.table_prep_params = config['table_prep_params']

    def generate_rand_cell(self, table, table_lst):
        def get_rand_table():
            rand_table_name = random.choice(range(len(self.all_tables)))
            t_d = self.all_tables[rand_table_name]
            return copy.deepcopy(t_d), rand_table_name

        table_data, table_name = get_rand_table()
        numDataRows, numCols = np.array(table_data).shape[:2]
        rand_row_ix = random.choice(list(range(numDataRows)))
        rand_col_ix = random.choice(list(range(numCols)))
        rand_cell = table_data[rand_row_ix][rand_col_ix]

        if len(rand_cell) == 0 and random.random() < 0.6:
            return self.generate_rand_cell(table, table_lst)

        # while (
        #     # True
        #     # rand_cell in flatten_1_deg(table) or
        #     # table_name in table_lst or
        #     rand_cell.count(1) == len(rand_cell)
        # ):
        #     # if table_name in table_lst:
        #     #     return generate_rand_cell(table, table_lst)
        #     # elif rand_cell.count(1) == len(rand_cell):
        #     #     rand_row_ix = random.choice(list(range(numDataRows)))
        #     #     rand_col_ix = random.choice(list(range(numCols)))
        #     #     rand_cell = table_data[rand_row_ix][rand_col_ix]
        #     # else:
        #     #     break
        #     # print('sample table again coz repeat')
        #     rand_cell, table_name = generate_rand_cell(table, table_lst)
        return rand_cell, table_name

    def generate_neg_table(self, inp):
        t_l = []
        t = []
        row_sh, col_sh = np.array(inp).shape[:2]
        self.table_prep_params = {
            'MAX_ROW_LEN': row_sh,
            'MAX_COL_LEN': col_sh,
        }
        for i in range(self.table_prep_params['MAX_ROW_LEN']):
            r = []
            for j in range(self.table_prep_params['MAX_COL_LEN']):
                c, t_name = self.generate_rand_cell(t, t_l)
                r.append(c)
                t_l.append(t_name)
            t.append(r)
        return t

    def generate_neg(self, vocab):
        size = len(self.all_tables)
        Xn = [self.generate_neg_table(table) for table in self.all_tables]
        yn = np.ones((size, 1)) * 0
        Xn, yn = T2VDataset(np.array(Xn), yn, vocab,
                            self.config, run_pipe=True).return_all()
        return Xn, yn


class T2VDataset(Dataset):
    def __init__(self, X, y, vocab, config, run_pipe=False):
        if run_pipe:
            X = self.pipe(X, vocab, config)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y.tolist(), dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx

    def pipe(self, X, vocab, config):
        X = X.tolist()
        pad_id = vocab.index('<PAD>')
        for i in range(len(X)):
            X[i] = T2VDataset.pad_table(
                config['table_prep_params'], X[i], pad_id)
        return X

    def return_all(self):
        return self.X, self.y

    @staticmethod
    def pad_table(table_prep_params, table, pad_id):
        t_ = np.array(table)
        rows, cols = t_.shape[:2]
        cols2fill = table_prep_params['MAX_COL_LEN'] - cols
        c_ = int(cols2fill / 2)
        rows2fill = table_prep_params['MAX_ROW_LEN'] - rows
        r_ = int(rows2fill / 2)

        full_t = np.full((
            table_prep_params['MAX_ROW_LEN'], table_prep_params['MAX_COL_LEN'], 1
        ), pad_id)
        full_t[r_:r_ + rows, c_:c_ + cols] = t_
        return full_t.tolist()


def collate_fn(batch, Xp_unpad, config, vocab):
    X_batch = torch.cat([i[0].unsqueeze(0) for i in batch])
    y_batch = torch.cat([i[1].unsqueeze(0) for i in batch])
    idx = [i[2] for i in batch]
    X_batch_unpad = Xp_unpad[idx]
    Xn, yn = NegSample(X_batch_unpad, config).generate_neg(vocab)
    total_inputs = torch.cat((X_batch, Xn), dim=0)
    y_batch_total = torch.cat((y_batch, yn), dim=0)
    shuffle = torch.randperm(len(total_inputs))
    total_inputs = total_inputs[shuffle]
    y_batch_total = y_batch_total[shuffle]
    return total_inputs, y_batch_total, idx


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pad_data_prep",
                        help="path for the scores", action='store_true')
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
        pad_id = vocab.index('<PAD>')
        for i in range(len(X)):
            X[i] = T2VDataset.pad_table(table_prep_params, X[i], pad_id)
        print('After padding')
        X = np.array(X)
        print(X[12])
        print(X.shape)

        temp = args.xp_file.split('.')
        temp[0] = temp[0] + '_pad_unk'
        savepkl(os.path.join(Xp_path, '.'.join(temp)), X)
    else:
        config = Config()
        device = torch.device(
            f"cuda:{1}" if torch.cuda.is_available() else 'cpu')
        dataset = T2VDataset(X, vocab, config)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        X_, y_ = next(iter(dataloader))
        print(X_.shape, y_.shape)

    print(time.time() - start)

    # if args.test:
    #     x = loadpkl(config['input_files']['Xp_path'])
    #     vocab = loadpkl(config['input_files']['vocab_path'])
    #     x_unpad = loadpkl(config['input_files']['Xp_unpad_path'])
    #     device = torch.device(f"cuda:{args.gpu}")
    #     logger.info(len(vocab))
    #     Xn, yn = NegSample(
    #         x_unpad[:64], config).generate_neg(vocab)
