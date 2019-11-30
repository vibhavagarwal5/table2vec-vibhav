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
    logger.info('NDCG score: \n')
    logger.info(command.read())
    command.close()


if __name__ == '__main__':
    path = './output/11_30_20_20_58/TREC_results/LTR_k5_'
    trec_path = '../trec_eval/trec_eval'
    query_file_path = '../global_data/qrels.txt'

    ndcg_pipeline(path, trec_path, query_file_path)
