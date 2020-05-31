import json
import math
import os
import random
import re
from collections import Counter
import matplotlib.pyplot as plt
from itertools import chain
from functools import reduce

import numpy as np
import pandas as pd
from utils import loadpkl

# file paths
ALL_TABLES_PATH_ORG = '../global_data/tables_redi2_1/'
OUTPUT_DIR = '../global_data/all_tables'
all_tables = os.listdir(OUTPUT_DIR)

MAX_COL_LEN = 10
MAX_ROW_LEN = 50
SAMPLE_PERC = 0.5
LENGTH_PER_CELL = 20


def read_table(table):
    if table.split('.')[-1] == 'json':
        table = table.split('.')[0]
    with open(os.path.join(OUTPUT_DIR, f"{table}.json"), 'r') as f:
        j = json.load(f)
    return j


def get_stats(all_tables):
    tables = [[], []]
    for j in all_tables:
        j = np.array(j)
        if j.size > 0:
            no_of_rows, no_of_cols = j.shape[:2]
        else:
            no_of_rows, no_of_cols = 0, 0
        tables[0].append(no_of_rows)
        tables[1].append(no_of_cols)
    tables[0].sort()
    tables[1].sort()
    return tables


def get_cell_stats(all_tables):
    cells = []
    for j in all_tables:
        cells.append(list(set(list(chain(*j)))))
    cells = list(set(list(chain(*cells))))
    return cells


def dataset_stats(all_tables):
    tables = get_stats(all_tables)
    cols = []
    for i in list(set(tables[1])):
        cols.append(len([k for k in tables[1] if k == i]))
    print(f"cols distribution: {cols}\n")
    plt.plot(cols)
    # plt.plot(range(8, 20), cols[8:20])
    plt.savefig('cols.png')

    plt.clf()
    plt.cla()
    plt.close()

    rows = []
    for i in list(set(tables[0])):
        rows.append(len([k for k in tables[0] if k == i]))
    print(f"rows distribution: {rows}\n")
    plt.plot(rows)
    # plt.plot(range(20, 50), rows[20:50])
    plt.savefig('rows.png')

    unique_cells = get_cell_stats(all_tables)
    total_cells_count = 0
    print(f"unique_cells: {len(unique_cells)}\n")
    cell_len_over_20 = len([i for i in unique_cells if len(i.split(' ')) > 20])
    print(f"cell_len_over_20: {cell_len_over_20}\n")
    total_cells = reduce(
        lambda x, y: x+y, list(map(lambda x: x[0]*x[1], list(zip(tables[0], tables[1])))))
    print(f"total_cells: {total_cells}\n")


if __name__ == "__main__":
    # baseline_f = pd.read_csv('../global_data/features.csv')
    # tables_subset_3k = list(baseline_f['table_id'])
    # tables_subset = list(
    #     set(tables_subset_3k+random.sample(all_tables, 20000)))

    tables_subset = loadpkl('./data/postive_tables_set.pkl')
    read_all_tables = [read_table(js)['data'] for js in tables_subset]
    dataset_stats(read_all_tables)

    vocab = loadpkl('./data/vocab_2D_10-50_complete.pkl')
    print(f'vocab: {len(vocab)}\n')
