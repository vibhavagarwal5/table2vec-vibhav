import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Pool
from torch.nn.init import xavier_normal_, xavier_uniform_
from torchsummary import summary

from preprocess import print_table
from utils import flatten_1_deg, loadpkl, savepkl

torch.multiprocessing.set_start_method('spawn', force=True)


class Table2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, device, config):
        super(Table2Vec, self).__init__()
        self.device = device
        self.table_prep_params = config['table_prep_params']
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = True
        xavier_normal_(self.embeddings.weight.data)

        self.conv1 = nn.Conv2d(embedding_dim, 128, (5, 4), 2, 0)
        self.mp1 = nn.MaxPool2d((2, 2), 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128*11*2, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_rand_table(self):
        rand_table_name = random.choice(range(len(self.all_tables)))
        t_d = self.all_tables[rand_table_name].tolist()
        # print(type(self.all_tables), type(t_d))

        # numDataRows, numCols = np.array(t_d).shape[:2]
        # while numDataRows == 0 or numCols == 0:
        #     # print('sample table again coz 0')
        #     t_d, rand_table_name = self.get_rand_table()
        return t_d, rand_table_name

    def generate_rand_cell(self, table, table_lst):
        table_data, table_name = self.get_rand_table()
        # print(type(table_data))
        numDataRows, numCols = np.array(table_data).shape[:2]
        rand_row_ix = random.choice(list(range(numDataRows)))
        rand_col_ix = random.choice(list(range(numCols)))
        rand_cell = table_data[rand_row_ix][rand_col_ix]
        # print(type(rand_cell))

        # while (
        #     # True
        #     # rand_cell in flatten_1_deg(table) or
        #     # table_name in table_lst or
        #     rand_cell.count(1) == len(rand_cell)
        # ):
        #     # if table_name in table_lst:
        #     #     return self.generate_rand_cell(table, table_lst)
        #     # elif rand_cell.count(1) == len(rand_cell):
        #     #     rand_row_ix = random.choice(list(range(numDataRows)))
        #     #     rand_col_ix = random.choice(list(range(numCols)))
        #     #     rand_cell = table_data[rand_row_ix][rand_col_ix]
        #     # else:
        #     #     break
        #     # print('sample table again coz repeat')
        #     rand_cell, table_name = self.generate_rand_cell(table, table_lst)

        return rand_cell, table_name

    def generate_neg_table(self, _):
        t_l = []
        t = []

        rand_rows = random.choice(
            list(range(3, self.table_prep_params['MAX_ROW_LEN'])))
        rand_cols = random.choice(
            list(range(2, self.table_prep_params['MAX_COL_LEN'])))
        # rand_rows = self.table_prep_params['MAX_ROW_LEN']
        # rand_cols = self.table_prep_params['MAX_COL_LEN']

        for i in range(rand_rows):
            r = []
            for j in range(rand_cols):
                c, t_name = self.generate_rand_cell(t, t_l)
                r.append(c)
                t_l.append(t_name)
            t.append(r)
        t = self.pad_table(t)
        # print(_)
        return t

    def pad_table(self, table):
        for row in table:
            for j in range(0, self.table_prep_params['MAX_COL_LEN']-len(row)):
                row.append([1]*self.table_prep_params['LENGTH_PER_CELL'])
        for i in range(0, self.table_prep_params['MAX_ROW_LEN']-len(table)):
            table.append([[1]*self.table_prep_params['LENGTH_PER_CELL']]
                         * self.table_prep_params['MAX_COL_LEN'])
        return table

    def generate_neg(self, size):
        Xn = [self.generate_neg_table(i) for i in range(size)]
        # with Pool(40) as p:
        #     Xn = [tqdm(p.imap(self.generate_neg_table, range(size)), total=size)]
        # p = Pool(processes=20)
        # Xn = p.map(self.generate_neg_table, range(size))
        # p.close()
        # p.join()

        yn = [0]*size
        Xn = torch.tensor(Xn, device=self.device)
        yn = torch.tensor(yn, dtype=torch.float, device=self.device)
        return Xn, yn.reshape((-1, 1))

    def forward(self, inputs, y_true):
        self.all_tables = inputs
        Xn, yn = self.generate_neg(len(inputs))
        total_inputs = torch.cat((inputs, Xn), dim=0)
        total_y = torch.cat((y_true, yn), dim=0)

        emb = self.embeddings(total_inputs)
        emb = emb.mean(3)
        emb = emb.permute(0, 3, 1, 2)

        x = self.conv1(emb)
        # x = self.bn1(x)
        x = self.mp1(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.drop2(x)
        x = F.relu(x)
        x = self.fc2(x)
        pred = torch.sigmoid(x)
        return pred, total_y
