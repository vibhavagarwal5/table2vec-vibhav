import numpy as np
import pandas as pd
import os
from itertools import zip_longest
import json
from multiprocessing import Pool
import math
import random
import pickle
from collections import Counter
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import tqdm

# import utils

# file paths
ALL_TABLES_PATH_ORG = '../global_data/tables_redi2_1/'
OUTPUT_DIR = '../global_data/all_tables'


def read_table(table):
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    return j


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


def clean_data(table_d, filter_fname):
    for row_id, row in enumerate(table_d):
        for col_id, col in enumerate(row):
            table_d[row_id][col_id] = filter_fname(table_d[row_id][col_id])
    return table_d


def filter_entities(inp):
    if len(inp):
        if inp[0] == '[' and inp[-1] == ']':
            inp = inp.split('|')[0][1:]
        return inp
    else:
        return inp


def split_data(data):
    data = np.array(data)
    row_shape = data.shape[0]
    column_shape = data.shape[1]

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

def get_rand_table():
    rand_table = random.choice(all_tables)
    j = read_table(rand_table)

    while j['numDataRows'] == 0 or j['numCols'] == 0:
        # print('sample table again coz 0')
        j, rand_table = get_rand_table()
    return j, rand_table


def generate_rand_cell(table, table_l):
    j, rand_table = get_rand_table()

    rand_row_ix = random.choice(list(range(j['numDataRows'])))
    rand_col_ix = random.choice(list(range(j['numCols'])))
    rand_cell = j['data'][rand_row_ix][rand_col_ix]

    while rand_cell in list(sum(table, [])) or rand_table in table_l:
        # print('sample table again coz repeat')
        rand_cell, rand_table = generate_rand_cell(table, table_l)

    return rand_cell, rand_table


def generate_neg_table(i):
    t_l = []
    t = []

    rand_rows = random.choice(list(range(2, MAX_ROW_LEN)))
    rand_cols = random.choice(list(range(2, MAX_COL_LEN)))

    for i in range(rand_rows):
        r = []
        for j in range(rand_cols):
            c, t_name = generate_rand_cell(t, t_l)
            r.append(c)
            t_l.append(t_name)
        t.append(r)
    return t


def tokenize_table(table):
    table = clean_data(table, filter_entities)
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            table[i][j] = tokenize_cell(cell)
    return table


def tokenize_cell(cell):
    doc = n1(cell)
    t = [token.orth_ for token in doc if not token.is_punct |
         token.is_space | token.is_stop]
    # t = [[k.text for k in tokenizer2(i)] for i in t]
    # t = list(sum(t, []))
    return t


def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[$&+,:;=?@#|<>^*()%!]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    return Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        # suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
    )


def check_overflow(X, y):
    X_new = []
    l_ind = []
    for ind, table in enumerate(X):
        l = []
        l_ = []
        for row in table:
            for cell in row:
                if len(cell) > LENGTH_PER_CELL:
                    temp = row
                    l_ind.append(ind)
                    l_.append(row)
                    l.append(cell_overflow_split(temp))
                    break
        for i in l_:
            table.remove(i)
        for i in l:
            for j in i:
                table.append(j)
        tables = [table[x:x+MAX_COL_LEN]
                  for x in range(0, len(table), MAX_COL_LEN)]
        X_new.append(tables)
        for i in range(len(tables)-1):
            y.insert(i, y[i])
    X_new = list(sum(X_new, []))
    return np.array(X_new), np.array(y).reshape((-1, 1))


def cell_overflow_split(row):
    out_row = []
    for cell in row:
        out_row.append([cell[x:x+LENGTH_PER_CELL]
                        for x in range(0, len(cell), LENGTH_PER_CELL)])
    output = [[i if i is not None else [] for i in element]
              for element in list(zip_longest(*out_row))]
    return output


def cell_overflow_cap(X):
    for table in X:
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                if len(cell) > LENGTH_PER_CELL:
                    table[i][j] = clean_cell(cell)


def clean_cell(cell):
    for i in cell:
        if len(i) == 1 or i.isdigit():
            cell.remove(i)
    return cell[:LENGTH_PER_CELL]


def word_distr(table):
    l_ = list(sum(table, []))
    return list(sum(l_, []))


def print_table(table):
    for row in table:
        print(row)
    print()


def loadpkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def savepkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pipeline(X, label_type):
    '''
    Breaking tables into chunks over max col and row length
    '''
    if label_type == '+':
        X = [split_overflow_table(read_table(table)) for table in X]
        print(f"Final: {np.array(X).shape}")
        X = list(sum(X, []))
        print(f"If +ve final is unrolled::: {len(X)}")
    else:
        print(f"-ve final::: {len(X)}")

    '''
    Tokenizing X data
    '''
    p = Pool(processes=60)
    X = p.map(tokenize_table, X)
    p.close()
    p.join()
    print(len(X))

    '''
    Managing the overflow of tokens per cell by either clipping or splitting
    '''
    cell_overflow_cap(X)

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

    return X, y

# Testing tables
# table-0614-640.json
# table-1225-209.json
# print_table(X[6705])


all_tables = os.listdir(OUTPUT_DIR)
# config = utils.load_config_args()
# MAX_COL_LEN = config['table_prep_params']['MAX_COL_LEN']
# MAX_ROW_LEN = config['table_prep_params']['MAX_ROW_LEN']
# SAMPLE_PERC = config['table_prep_params']['SAMPLE_PERC']
# LENGTH_PER_CELL = config['table_prep_params']['LENGTH_PER_CELL']
MAX_COL_LEN = 10
MAX_ROW_LEN = 50
SAMPLE_PERC = 0.5
LENGTH_PER_CELL = 20

n1 = spacy.load("en_core_web_md")
# tokenizer2 = custom_tokenizer(spacy.load("en"))

if __name__ == "__main__":
    baseline_f = pd.read_csv('../global_data/features.csv')
    tables_subset = list(baseline_f['table_id'])

    # '''
    # Generating Positive X, y dataset from the tables
    # '''
    # X_p, y_p = pipeline(tables_subset, '+')
    # savepkl(f'./data/xp_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', X_p)
    # savepkl(f'./data/yp_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', y_p)

    # '''
    # Generating Negative X, y dataset from the tables
    # '''

    # p = Pool(processes=40)
    # X_n = p.map(generate_neg_table, range(int(len(X_p)*1.5)))
    # p.close()
    # p.join()

    # X_n, y_n = pipeline(X_n, '-')
    # savepkl(f'./data/xn_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', X_n)
    # savepkl(f'./data/yn_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', y_n)

    '''
    Generating word distribution dataframe
    '''
    X_n = loadpkl('./data/xn_2D_10-50.pkl')

    X_p = [read_table(table)['data'] for table in tables_subset]
    p = Pool(processes=30)
    result = p.map(tokenize_table, X_p)
    X_p = np.array(result)
    p.close()
    p.join()
    print(X_p.shape)

    X = np.hstack((X_n, X_p))
    p = Pool(processes=30)
    result = p.map(word_distr, X)
    p.close()
    p.join()
    result = list(sum(result, []))
    query_l = [tokenize_cell(i) for i in list(baseline_f['query'].unique())]
    query_l = list(sum(query_l, []))
    result+=query_l
    print(result[:10])
    count = Counter(result)
    c = []
    for i in count.keys():
        c.append([i, count[i]])
    df = pd.DataFrame(c)
    df.sort_values(by=[1], ascending=False, inplace=True)
    df.to_csv('./word_distr_2D_complete.csv', index=False, columns=None)

    '''
    Getting the vocab from the data
    '''
    vocab = list(set(count.keys()))
    vocab.insert(0, '<PAD>')
    vocab.insert(0, '<UNK>')
    print(len(vocab))
    savepkl(
        f'./data/vocab_2D_{MAX_COL_LEN}-{MAX_ROW_LEN}_complete.pkl', vocab)
