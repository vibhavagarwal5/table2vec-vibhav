import json
import math
import os
import random
import re
import unicodedata
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from utils import flatten_1_deg, loadpkl, savepkl, pool_fn

nlp = spacy.load("en_core_web_md")
nlp.add_pipe(nlp.create_pipe("merge_entities"))
nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))

OUTPUT_DIR = '../global_data/all_tables'
all_tables = os.listdir(OUTPUT_DIR)
PATH = './data/wo_strnum3.0_wo_ent_'
MAX_ROW_LEN = 15
MAX_COL_LEN = 5


def read_table(table):
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    return j


def clean_entities(inp):
    if len(inp):
        if inp[0] == '[' and inp[-1] == ']':
            inp = inp.split('|')[0][1:]  # Take the 1st element
            # inp = inp.split('|')[-1][:-1]  # Take the 2nd element
        return inp
    else:
        return inp


def tokenize_table(table):
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            table[i][j] = tokenize_str(clean_entities(cell))
    # table = filter_empty_cols(table)
    return table


def tokenize_str(cell):
    a = unicodedata.normalize('NFKD', cell).encode(
        'ascii', 'ignore').decode('utf-8')
    t = [token.orth_ for token in nlp(a) if not (
        token.is_punct
        or len(token.orth_) < 4
        or token.is_space
        or token.is_stop
        or token.is_currency
        or token.like_url
        or token.like_email
        or token.like_num
        or small_alphanum(token.orth_)
        or token.ent_type_ in ['DATE', 'TIME', 'MONEY', 'PERCENT']
    )]
    t = [i.replace(" ", "_") for i in t]
    return t


def small_alphanum(s):
    if len([i for i in s if i.isalpha()]) < 3:
        return True
    else:
        return False


def clean(table):
    to_rem = ['=', '{', '}', '</', ':#', '\\\\', '\\', '3px']
    to_rep_dash = ['(', '),', ')', '&amp', ',_', ':_', '/_', '+_']
    to_rep_sp = ['|', '?', ':', '#', '~', '$', '^', '\\n', ';', '@']
    to_rep_dash_rgx = "\(|\),|,_|\)|&amp|:_|/_|\+_"
    to_rep_sp_rgx = "\||\?|\:|#|~|\$|\^|\\n|;|@"

    def clean_cell(cell):
        tmp = []
        for w in cell[:]:
            if any(c in w for c in to_rem):
                cell.remove(w)
            elif any(c in w for c in to_rep_dash) or any(c in w for c in to_rep_sp):
                if any(c in w for c in to_rep_sp):
                    t = re.sub(to_rep_sp_rgx, " ", re.sub(
                        to_rep_dash_rgx, "_", w))
                else:
                    t = re.sub(to_rep_dash_rgx, "_", w)
                nw = ' '.join(list(filter(None, re.split(" _|_ ", t))))
                nw = '_'.join(list(filter(None, re.split("_", nw))))
                tmp.append(nw)
                cell.remove(w)
        cell_ = cell + tmp
        return tokenize_str(" ".join(list(dict.fromkeys(cell_))))

    for row in table:
        for i, cell in enumerate(row):
            row[i] = clean_cell(cell)
    return table


def remove_empty_tables(tables):
    e_t = []
    for i in range(len(tables)):
        if np.array(tables[i]).size == 0:
            e_t.append(i)
    return np.delete(tables, e_t)


def remove_empty_cols(table):
    def check_cell_validity(column):
        c = 0
        for i in column:
            if len(i) == 0:
                c += 1
        r = c / len(column)
        if r == 1:
            return True
#         elif r >= 0.7 and len(column)>4:
#             return True
        return False

    data = np.array(table)
    col = 0
    while(col < data.shape[1]):
        if check_cell_validity(data[:, col]):
            data = np.delete(data, col, 1)
        else:
            col += 1
    return data.tolist()


def remove_empty_rows(table):
    def check_cell_validity(row):
        c = 0
        for i in row:
            if len(i) == 0:
                c += 1
        r = c / len(row)
        if r == 1:
            return True
        return False

    data = np.array(table)
    row = 0
    while(row < data.shape[0]):
        if check_cell_validity(data[row, :]):
            data = np.delete(data, row, 0)
        else:
            row += 1
    return data.tolist()


def remove_dupl_rows(table):
    t = []
    for row in table:
        if row not in t:
            t.append(row)
    return t


def remove_dupl_cols(table):
    table = np.array(table)
    if len(table.shape) == 3:
        table_t = np.transpose(table, (1, 0, 2)).tolist()
    elif len(table.shape) == 2:
        table_t = np.transpose(table, (1, 0)).tolist()
    t = []
    for row in table_t:
        if row not in t:
            t.append(row)
    if len(table.shape) == 3:
        f_table = np.transpose(np.array(t), (1, 0, 2)).tolist()
    elif len(table.shape) == 2:
        f_table = np.transpose(np.array(t), (1, 0)).tolist()
    return f_table


def remove_1x1_table(X):
    ts_1 = []
    for i in range(len(X)):
        if np.array(X[i]).shape[:2] == (1, 1):
            ts_1.append(i)
    return np.delete(X, ts_1)


def split_data(data):
    data = np.array(data)
    (row_shape, column_shape) = data.shape[:2]

    blocks_per_row = math.ceil(row_shape / MAX_ROW_LEN)
    blocks_per_column = math.ceil(column_shape / MAX_COL_LEN)
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
    numDataRows, numCols = np.array(j).shape[:2]

    if numCols > MAX_COL_LEN or numDataRows > MAX_ROW_LEN:
        # print('Splitting the data')
        splits = split_data(j)
        for v in splits:
            if v.size != 0:
                # print('Adding split data')
                X.append(v.tolist())
    else:
        X.append(j)
    return X


def shrink_cell_len(table):
    for row in table:
        for i, cell in enumerate(row):
            if len(cell) > 1:
                row[i] = [cell[-1]]
    return table


def generate_vocab(X):
    baseline_f = pd.read_csv('../global_data/features.csv')
    result = flatten_1_deg(flatten_1_deg(flatten_1_deg(X.tolist())))
    print(f"table only vocab: {len(result)}, {len(list(set(result)))}")
    query_l = [tokenize_str(i.lower())
               for i in list(baseline_f['query'].unique())]
    query_l = flatten_1_deg(query_l)
    result += query_l
    # print(result[:10])
    count = Counter(result)
    c = [[i, count[i]] for i in count.keys()]
    df = pd.DataFrame(c)
    df.sort_values(by=[1], ascending=False, inplace=True)
    df.to_csv(f'{PATH}/word_distr.csv', index=False, columns=None)

    vocab = list(set(count.keys()))
    vocab.insert(0, '<PAD>')
    vocab.insert(0, '<UNK>')
    print(f'total vocab: {len(vocab)}\n')
    savepkl(
        f'{PATH}/vocab_{MAX_COL_LEN}-{MAX_ROW_LEN}.pkl', vocab)


def table_shape_stats(X):
    t_sh = []
    for table in X:
        t_sh.append(np.array(table).shape[:2])
    print(f"Total shapes: {len(t_sh)}, unqiue: {len(list(set(t_sh)))}\n")

    sh_distr = Counter(t_sh)
    t_s = list(sh_distr.keys())
    t_s_i = list(range(len(t_s)))
    t_s_val = [sh_distr[i] for i in sh_distr.keys()]

    sh_distr = sorted(list(zip(t_s, t_s_val)),
                      key=lambda x: x[1], reverse=True)
    print(f"Shape distribution: {sh_distr}\n")
    return t_sh, t_s, t_s_i, t_s_val


def get_avg_table_sh(X):
    t_sh, t_s, t_s_i, t_s_val = table_shape_stats(X)
    r = sum([a * b for a, b in list(zip([x[0]
                                         for x in t_s], t_s_val))]) / len(t_sh)
    c = sum([a * b for a, b in list(zip([x[1]
                                         for x in t_s], t_s_val))]) / len(t_sh)
    print(f"shape: {r}  x  {c}")


def cell_stats(X):
    all_cells = flatten_1_deg(flatten_1_deg(X.tolist()))
    print(
        f"Total cells: {len(all_cells)}, unqiue: {len(list(map(list, set(map(lambda i: tuple(i), all_cells)))))}\n")

    all_cells_len = list(map(lambda i: len(i), all_cells))
    cell_len_distr = Counter(all_cells_len)
    cell_len_distr = sorted(cell_len_distr.items(), key=lambda i: i[0])
    c_len, c_len_val = list(zip(*cell_len_distr))

    print(f"cell_len_distr: {cell_len_distr}")
    return all_cells, all_cells_len, c_len, c_len_val, cell_len_distr


def remove_emptiness(X):
    X = remove_empty_tables(X)
    print(X.shape)
    for i in range(len(X)):
        X[i] = remove_empty_cols(X[i])
    print(X.shape)
    X = remove_empty_tables(X)
    print(X.shape)

    for i in range(len(X)):
        X[i] = remove_empty_rows(X[i])
    print(X.shape)
    X = remove_empty_tables(X)
    print(X.shape)

    for i in range(len(X)):
        X[i] = remove_dupl_cols(X[i])
    print(X.shape)
    X = remove_empty_tables(X)
    print(X.shape)

    for i in range(len(X)):
        X[i] = remove_dupl_rows(X[i])
    print(X.shape)
    X = remove_empty_tables(X)
    print(X.shape)
    return X

# # Rejoining and splitting for entity check

# def retokenize2merge_ent(table):
#     for row in table:
#         for i, cell in enumerate(row):
#             if len(cell)>1:
#                 row[i] = tokenize_str(" ".join(list(dict.fromkeys(cell))))
#     return table


if __name__ == '__main__':
    baseline_f = pd.read_csv('../global_data/features.csv')
    tables_subset_3k = list(baseline_f['table_id'])
    tables_subset = list(set(
        tables_subset_3k + random.sample(all_tables, 20000)
    ))
    savepkl(f'{PATH}/postive_tables_set.pkl', tables_subset)
    read_all_tables = [read_table(js)['data'] for js in tables_subset]

    print(len(read_all_tables))

    print('---Tokenizing---\n\n')
    X = pool_fn(tokenize_table, read_all_tables, 75)
    X = np.array(X)
    print(X.shape)
    savepkl(f'{PATH}/x_tokenised.pkl', X)

    print("---Cleaning for some spl characters and patterns and merging back tokens to reduce token space---\n\n")
    X = pool_fn(clean, X, 75)
    X = np.array(X)
    X = remove_emptiness(X)

    print('---Splitting into smaller blocks---\n\n')
    print(X.shape)
    X = [split_overflow_table(table) for table in X.tolist()]
    X = flatten_1_deg(X)
    X = np.array(X)
    print(X.shape)

    X = remove_emptiness(X)
    # get_avg_table_sh(X)

    # print(X.shape)
    # X = remove_1x1_table(X)
    # print(X.shape)

    for i, table in enumerate(X):
        X[i] = shrink_cell_len(table)
    savepkl(f'{PATH}/x_tokenised_preprocessed.pkl', X)

    print("---Generating Vocab---\n\n")
    generate_vocab(X)

    # all_cells, all_cells_len, c_len, c_len_val, cell_len_distr = cell_stats(X)
    # sum([a * b for a, b in cell_len_distr[1:]]) / \
    #     (len(all_cells) - cell_len_distr[0][1])

    # all_words = flatten_1_deg(all_cells)
    # len(all_words), len(list(set(all_words)))
