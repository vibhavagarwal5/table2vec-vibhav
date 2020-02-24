import json
import math
import os
import random
import re
from collections import Counter
from itertools import chain, zip_longest
from multiprocessing import Pool

import numpy as np
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from utils import loadpkl, savepkl, flatten_1_deg
from data_viz import dataset_stats

# file paths
ALL_TABLES_PATH_ORG = '../global_data/tables_redi2_1/'
OUTPUT_DIR = '../global_data/all_tables'
all_tables = os.listdir(OUTPUT_DIR)

MAX_COL_LEN = 10
MAX_ROW_LEN = 50
SAMPLE_PERC = 0.5
LENGTH_PER_CELL = 20

nlp = spacy.load("en_core_web_md")


def read_table(table):
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    return j


def clean_entities(inp):
    if len(inp):
        if inp[0] == '[' and inp[-1] == ']':
            # inp = inp.split('|')[0][1:]
            inp = inp.split('|')[-1][:-1]
        return inp
    else:
        return inp


def split_data(data):
    data = np.array(data)
    (row_shape, column_shape) = data.shape

    blocks_per_row = math.ceil(row_shape/MAX_ROW_LEN)
    blocks_per_column = math.ceil(column_shape/MAX_COL_LEN)
    previous_row = 0
    for row_block in range(blocks_per_row):
        previous_row = row_block * MAX_ROW_LEN
        previous_column = 0
        for column_block in range(blocks_per_column):
            previous_column = column_block * MAX_COL_LEN
            block = data[previous_row:previous_row + MAX_ROW_LEN,
                         previous_column:previous_column + MAX_COL_LEN]
            yield block


def split_overflow_table(j):
    X = []
    if j['numCols'] == 0 or j['numDataRows'] == 0:
        # print('Empty tables')
        return X

    if j['numCols'] > MAX_COL_LEN or j['numDataRows'] > MAX_ROW_LEN:
        # print('Splitting the data')
        splits = split_data(j['data'])
        for v in splits:
            if v.shape[0] != 0 or v.shape[1] != 0:
                # print('Adding split data')
                X.append(v.tolist())
    else:
        X.append(j['data'])

    return X


def tokenize_table(table):
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            table[i][j] = tokenize_str(clean_entities(cell))
    table = filter_empty_cols(table)
    return table


def tokenize_str(cell):
    t = [token.orth_ for token in nlp(cell) if not (
        token.is_punct
        or token.is_space
        or token.is_stop
        or token.like_num
        or contains_num(token.orth_)
        or token.is_currency
        or token.orth_ == '>'
        or token.orth_ == '<'
        or len(token.orth_) < 4
    )]
    return t


def contains_num(s):
    return any(c.isdigit() for c in s)


def cell_overflow_cap(X):
    def clip(cell):
        if len(cell) > LENGTH_PER_CELL:
            return cell[:LENGTH_PER_CELL]
        else:
            return cell

    for t, table in enumerate(X):
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                table[i][j] = clip(cell)
    return X


# def column_filter(table):
#     def check_cell_validity(column):
#         c = 0
#         for i in column:
#             if len(i) < 4:
#                 c += 1
#             if contains_num(i):
#                 return True
#         if c/len(column) > 0.3:
#             return True
#         return False

#     data = np.array(table)
#     cols = data.shape[1]
#     col = 0
#     while(col < data.shape[1]):
#         if check_cell_validity(data[:, col]):
#             data = np.delete(data, col, 1)
#         else:
#             col += 1
#     return data.tolist()


def filter_empty_cols(table):
    def check_cell_validity(column):
        c = 0
        for i in column:
            if len(i) == 0:
                c += 1
        r = c/len(column)
        if len(column) < 20 and r >= 0.33:
            return True
        elif len(column) >= 20 and c/len(column) >= 0.5:
            return True
        return False

    data = np.array(table)
    cols = data.shape[1]
    col = 0
    while(col < data.shape[1]):
        if check_cell_validity(data[:, col]):
            data = np.delete(data, col, 1)
        else:
            col += 1
    return data.tolist()


def print_table(table):
    for row in table:
        for col in row:
            print(col)
        print()


def remove_empty_tables(all_tables):
    for i, t in enumerate(all_tables):
        if np.array(t).size == 0:
            del all_tables[i]
    return all_tables


def generate_vocab(X):
    '''
    Generating word distribution dataframe
    '''
    result = flatten_1_deg(flatten_1_deg(flatten_1_deg(X)))
    query_l = [tokenize_str(i) for i in list(baseline_f['query'].unique())]
    query_l = flatten_1_deg(query_l)
    result += query_l
    # print(result[:10])
    count = Counter(result)
    c = [[i, count[i]] for i in count.keys()]
    df = pd.DataFrame(c)
    df.sort_values(by=[1], ascending=False, inplace=True)
    df.to_csv('./data/word_distr_2D_complete.csv', index=False, columns=None)

    '''
    Getting the vocab from the data
    '''
    vocab = list(set(count.keys()))
    vocab.insert(0, '<PAD>')
    vocab.insert(0, '<UNK>')
    print(f'vocab: {len(vocab)}\n')
    savepkl(
        f'./data/vocab_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}_complete.pkl', vocab)


def data_prep_pipeline(X):
    '''
    Breaking tables into chunks over max col and row length
    '''
    X = [split_overflow_table(read_table(table)) for table in X]
    print(f"Intial # of tables before splitting::: {len(X)}")
    X = flatten_1_deg(X)
    print(f"Final # of tables::: {len(X)}")

    '''
    Tokenizing and filtering out empty columns from X
    '''
    # with Pool(50) as p:
    #     X = [tqdm(p.imap(tokenize_table, X), total=len(X))]
    p = Pool(processes=50)
    X = p.map(tokenize_table, X)
    p.close()
    p.join()
    # print_table(X[0])

    '''
    Remove totally empty tables, generating vocab and cliiping cells to max_cell_len
    '''
    generate_vocab(X)
    X = remove_empty_tables(X)
    X = cell_overflow_cap(X)

    y = [1]*len(X)
    y = np.array(y).reshape(-1, 1)
    X = np.array(X)
    print(X.shape, y.shape)
    print_table(X[0])

    return X, y


if __name__ == "__main__":
    baseline_f = pd.read_csv('../global_data/features.csv')
    tables_subset_3k = list(baseline_f['table_id'])
    tables_subset = list(
        set(tables_subset_3k+random.sample(all_tables, 20000)))

    # column_filter(read_table(tables_subset[0])['data'])

    savepkl('./data/postive_tables_set.pkl', tables_subset)
    read_all_tables = [read_table(js)['data'] for js in tables_subset]
    dataset_stats(read_all_tables)

    '''
    Generating Positive X, y dataset from the tables
    '''
    X, y = data_prep_pipeline(tables_subset)
    savepkl(f'./data/xp_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', X)
    savepkl(f'./data/yp_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', y)
