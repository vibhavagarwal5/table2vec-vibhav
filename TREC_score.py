import argparse
import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger("app")


def ndcg_pipeline(path, trec_path, query_file_path):
    d = pd.DataFrame()
    for i in range(5):
        d_ = pd.read_csv(f"{path}{i}.txt", sep=' ', header=None)
        d_.drop(columns=[3], inplace=True)
        d = pd.concat([d, d_])

    d_sorted = d.sort_values(by=[0, 4], ascending=[True, False])

    d_sorted_filtered = pd.DataFrame()
    for i in range(1, 61):
        d_sorted_filtered = pd.concat(
            [d_sorted_filtered, d_sorted[d_sorted[0] == i].iloc[:20]])

    l = []
    for i in range(60):
        for j in range(1, 21):
            l.append(j)
    d_sorted_filtered[3] = l
    d_sorted_filtered = d_sorted_filtered[list(range(6))]
    d_sorted_filtered.to_csv(
        f'{path}all.txt', sep=' ', index=False, header=False)

    command = os.popen(
        f"{trec_path} -m ndcg_cut.5,10,15,20 {query_file_path} {path}all.txt")
    result = command.read()
    command.close()
    return result, get_ndcg_dict(result)


def get_ndcg_dict(ndcg_str):
    d = {}
    for i in ndcg_str.strip().split('\n'):
        l = i.split('\tall\t')
        d[l[0].strip()] = float(l[1])
    return d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="path for the scores")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    path = f'./output/{args.path}/TREC_results/LTR_k5_'
    trec_path = '../trec_eval/trec_eval'
    query_file_path = '../global_data/qrels.txt'

    res, res_d = ndcg_pipeline(path, trec_path, query_file_path)
    print(res, res_d)
