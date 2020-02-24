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
from spacy.util import (compile_infix_regex, compile_prefix_regex,
                        compile_suffix_regex)
from tqdm import tqdm

from utils import loadpkl, savepkl, flatten_1_deg

# file paths
ALL_TABLES_PATH_ORG = '../global_data/tables_redi2_1/'
OUTPUT_DIR = '../global_data/all_tables'
all_tables = os.listdir(OUTPUT_DIR)

MAX_COL_LEN = 10
MAX_ROW_LEN = 50
SAMPLE_PERC = 0.5
LENGTH_PER_CELL = 20

nlp = spacy.load("en_core_web_md")
# tokenizer2 = custom_tokenizer(spacy.load("en"))


# -----------------------------------------------------------------------------------------------


# Read the tables content
def read_table(table):
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    return j


# def clean_data(table_d, filter_fname):
#     for row_id, row in enumerate(table_d):
#         for col_id, col in enumerate(row):
#             table_d[row_id][col_id] = filter_fname(table_d[row_id][col_id])
#     return table_d


# Cleaning proper entities into words eg, [a_b|a b] -> "a b"
def clean_entities(inp):
    if len(inp):
        if inp[0] == '[' and inp[-1] == ']':
            # inp = inp.split('|')[0][1:]
            inp = inp.split('|')[-1][:-1]
        return inp
    else:
        return inp


# Helper function for splitting tables into smaller blocks
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


# Splitting a large table over MAX_ROW_LEN x MAX_COL_LEN into smaller blocks of these sizes
def split_overflow_table(j):
    X = []
    if j['numCols'] == 0 or j['numDataRows'] == 0:
        print('Empty tables')
        return X

    if j['numCols'] > MAX_COL_LEN or j['numDataRows'] > MAX_ROW_LEN:
        print('Splitting the data')
        splits = split_data(j['data'])
        for v in splits:
            if v.shape[0] != 0 or v.shape[1] != 0:
                print('Adding split data')
                X.append(v.tolist())
    else:
        X.append(j['data'])

    return X


# def generate_rand_table(table, no_sample):
#     rand_table = random.choice(all_tables)
#     if rand_table.split('.')[-1] == 'json':
#         rand_table = rand_table.split('.')[0]

#     print('Getting the random table')
#     with open(os.path.join(OUTPUT_DIR, f"{rand_table}.json"), 'r') as f:
#         j = json.load(f)

#     while j['data'] == table or j['numDataRows']*j['numCols'] <= no_sample:
#         print('sample table again')
#         j = generate_rand_table(table, no_sample)
#     return j


# def generate_neg(table, inp_t):
#     inp_t = np.array(inp_t)
#     row_len = inp_t.shape[0]
#     col_len = inp_t.shape[1]

#     no_sample = math.ceil(SAMPLE_PERC * (row_len + col_len))

#     r = list(range(row_len))
#     random.shuffle(r)
#     random_r_ixs = r[:no_sample]
#     c = list(range(col_len))
#     random.shuffle(c)
#     random_c_ixs = c[:no_sample]

#     j = generate_rand_table(table, no_sample)

#     print('Getting the random value from the table')
#     rand_row_ix = random.choice(list(range(j['numDataRows'])))
#     rand_col_ix = random.choice(list(range(j['numCols'])))
#     rand_val = j['data'][rand_row_ix][rand_col_ix]
#     c = 0
#     while rand_val in (text for rw in table for text in rw) and c <= no_sample:
#         print('sample value again')
#         rand_row_ix = random.choice(list(range(j['numDataRows'])))
#         rand_col_ix = random.choice(list(range(j['numCols'])))
#         rand_val = j['data'][rand_row_ix][rand_col_ix]
#         c += 1

#     if c > no_sample:
#         return generate_neg(table, inp_t)
#     else:
#         for i, j in zip(random_r_ixs, random_c_ixs):
#             inp_t[i][j] = rand_val
#         return inp_t.tolist()


# Getting random tables from the ~1.5million dataset
def get_rand_table():
    rand_table = random.choice(all_tables)
    j = read_table(rand_table)

    while j['numDataRows'] == 0 or j['numCols'] == 0:
        # print('sample table again coz 0')
        j, rand_table = get_rand_table()
    return j, rand_table


# Getting random cells from the random table picked
def generate_rand_cell(table, table_l):
    j, rand_table = get_rand_table()

    rand_row_ix = random.choice(list(range(j['numDataRows'])))
    rand_col_ix = random.choice(list(range(j['numCols'])))
    rand_cell = j['data'][rand_row_ix][rand_col_ix]

    while rand_cell in flatten_1_deg(table) or rand_table in table_l:
        # print('sample table again coz repeat')
        rand_cell, rand_table = generate_rand_cell(table, table_l)

    return rand_cell, rand_table


# Generating random table of random rows and columns
def generate_neg_table(_):
    t_l = []
    t = []

    rand_rows = random.choice(list(range(3, MAX_ROW_LEN)))
    rand_cols = random.choice(list(range(2, MAX_COL_LEN)))

    for i in range(rand_rows):
        r = []
        for j in range(rand_cols):
            c, t_name = generate_rand_cell(t, t_l)
            r.append(c)
            t_l.append(t_name)
        t.append(r)
    return t


# tokenizing table cells
# NOTE: MAKE THIS FAST.....
def tokenize_table(table):
    # data = np.array(table)
    # if len(data.shape) < 2:
    #     return table
    # (no_of_rows, no_of_cols) = data.shape
    # # data = data.reshape((no_of_rows*no_of_cols))
    # data = data.flatten()
    # data = np.vectorize(clean_entities)(data)
    # # data = list(nlp.pipe(data.tolist(), batch_size=50, n_process=15))
    # data = [tokenize_str(cell) for cell in data.tolist()]
    # data = np.array(data)
    # data = data.reshape((no_of_rows, no_of_cols))
    # return data.tolist()
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            table[i][j] = tokenize_str(clean_entities(cell))
    return table


# String tokenization
def tokenize_str(cell):
    # cell = np.array(nlp(cell))
    # cell = np.vectorize(is_not_valid, otypes=[str])(cell)
    # t = cell.tolist()
    t = [token.orth_ for token in nlp(cell) if not (token.is_punct or token.is_space or token.is_stop or token.like_num or contains_num(
        token.orth_) or token.is_currency or token.orth_ == '>' or token.orth_ == '<')]
    # # t = list(sum([[k.text for k in tokenizer2(i)] for i in t], []))
    return t


# def is_not_valid(token):
#     if not (token.is_punct or token.is_space or token.is_stop or token.like_num or contains_num(token.orth_) or token.is_currency or token.orth_ == '>' or token.orth_ == '<'):
#         return token


# If a string contains a digit or not
def contains_num(s):
    return any(c.isdigit() for c in s)


# def custom_tokenizer(nlp):
#     infix_re = re.compile(r'''[$&+,:;=?@#|<>^*()%!]''')
#     prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
#     suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
#     return Tokenizer(
#         nlp.vocab,
#         prefix_search=prefix_re.search,
#         # suffix_search=suffix_re.search,
#         infix_finditer=infix_re.finditer,
#     )


# def check_overflow(X, y):
#     X_new = []
#     l_ind = []
#     for ind, table in enumerate(X):
#         l = []
#         l_ = []
#         for row in table:
#             for cell in row:
#                 if len(cell) > LENGTH_PER_CELL:
#                     temp = row
#                     l_ind.append(ind)
#                     l_.append(row)
#                     l.append(cell_overflow_split(temp))
#                     break
#         for i in l_:
#             table.remove(i)
#         for i in l:
#             for j in i:
#                 table.append(j)
#         tables = [table[x:x+MAX_COL_LEN]
#                   for x in range(0, len(table), MAX_COL_LEN)]
#         X_new.append(tables)
#         for i in range(len(tables)-1):
#             y.insert(i, y[i])
#     X_new = list(sum(X_new, []))
#     return np.array(X_new), np.array(y).reshape((-1, 1))


# def cell_overflow_split(row):
#     out_row = []
#     for cell in row:
#         out_row.append([cell[x:x+LENGTH_PER_CELL]
#                         for x in range(0, len(cell), LENGTH_PER_CELL)])
#     output = [[i if i is not None else [] for i in element]
#               for element in list(zip_longest(*out_row))]
#     return output


# Clipping overflowing cells with more than 20 token to 20 tokens/cell
def cell_overflow_cap(X):
    def clip(cell):
        if len(cell) > LENGTH_PER_CELL:
            return cell[:LENGTH_PER_CELL]
        else:
            return cell

    for t, table in enumerate(X):
        # data = np.array(table)
        # data = np.vectorize(clip, otypes=[list])(data)
        # X[t] = data.tolist()

        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                table[i][j] = clip(cell)
    return X


def print_table(table):
    for row in table:
        for col in row:
            print(col)
        print()


#  Main data clean and preparation pipeline
def data_prep_pipeline(X, label_type):
    '''
    Breaking tables into chunks over max col and row length
    '''
    if label_type == '+':
        X = [split_overflow_table(read_table(table)) for table in X]
        print(f"Final: {np.array(X).shape}")
        X = flatten_1_deg(X)
        print(f"If +ve final is unrolled::: {len(X)}")
    else:
        print(f"-ve final::: {len(X)}")

    '''
    Tokenizing X data
    '''
    with Pool(50) as p:
        X = [tqdm(p.imap(tokenize_table, X), total=len(X))]
    # p = Pool(processes=20)
    # X = p.map(tokenize_table, X)
    p.close()
    p.join()
    # print_table(X[0])

    # '''
    # Managing the overflow of tokens per cell by either clipping or splitting
    # '''
    # X = cell_overflow_cap(X)  # Clipping

    '''
	Generate y labels
	'''
    if label_type == '+':
        y = [1]*len(X)
    elif label_type == '-':
        y = [0]*len(X)
    y = np.array(y).reshape(-1, 1)
    X = np.array(X)
    print(X.shape, y.shape)
    print_table(X[0])

    return X, y

# Testing tables
# table-0614-640.json
# table-1225-209.json
# print_table(X[6705])


if __name__ == "__main__":
    baseline_f = pd.read_csv('../global_data/features.csv')
    tables_subset_3k = list(baseline_f['table_id'])
    tables_subset = list(
        set(tables_subset_3k+random.sample(all_tables, 20000)))

    '''
    Generating Positive X, y dataset from the tables
    '''
    X_p, y_p = data_prep_pipeline(tables_subset, '+')
    savepkl(f'./data/xp_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', X_p)
    savepkl(f'./data/yp_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', y_p)

    '''
    Generating Negative X, y dataset from the tables (1.3 x X_p)
    '''
    size = int(len(X_p)*1.3)
    with Pool(40) as p:
        X_n = [tqdm(p.imap(generate_neg_table, range(size)), total=size)]
    p.close()
    p.join()
    X_n, y_n = data_prep_pipeline(X_n, '-')
    savepkl(f'./data/xn_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', X_n)
    savepkl(f'./data/yn_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', y_n)

    '''
    Generating word distribution dataframe
    '''
    X_n = loadpkl('./data/xn_2D_10-50.pkl')
    X_p = loadpkl('./data/xp_2D_10-50.pkl')
    X = np.hstack((X_n, X_p))
    print(X.shape, X_n.shape, X_p.shape)
    result = flatten_1_deg(flatten_1_deg(flatten_1_deg(X)))
    print('Adding queries...')
    query_l = [tokenize_str(i) for i in list(baseline_f['query'].unique())]
    query_l = flatten_1_deg(query_l)
    result += query_l
    print(result[:10])
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
    print(len(vocab))
    savepkl(
        f'./data/vocab_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}_complete.pkl', vocab)
