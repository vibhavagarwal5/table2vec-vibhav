import numpy as np
import pandas as pd
import os
import torch
from preprocess import loadpkl, print_table, savepkl, split_overflow_table, tokenize_table, read_table, tokenize_cell
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=15, shm_size_mb=3000)


def get_emb(table):
    table_ = table[:]
    for row in table_:
        for j, cell in enumerate(row):
            for i, item in enumerate(cell):
                cell[i] = embeddings[w2i[item]]
            row[j] = np.average(row[j], axis=0).tolist()
    return table_


def late_fusion(table, query):
    s = []
    for i in query:
        for j in table:
            sim = cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))
            s.append(sim)
    s = np.array(s).reshape(-1)
    return s


def early_fusion(table, query):
    a = np.average(table, axis=0).reshape(1, -1)
    b = np.average(query, axis=0).reshape(1, -1)
    sim = cosine_similarity(a, b)
    return sim.reshape(-1)[0]


def mp(df, func, num_partitions):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_partitions)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def t(df):
    df['table_emb'] = df.table_id.apply(
        lambda x: get_emb(tokenize_table(read_table(x)['data'])))
    return df


if __name__ == '__main__':

    # X = loadpkl('./data/xp_2D_10-50.pkl')
    vocab = loadpkl('./data/vocab_2D_10-50_complete.pkl')

    w2i = {w: i for i, w in enumerate(vocab)}

    model = torch.load('./output/11_25_15_56_30/model.pt')
    embeddings = model['embeddings.weight'].cpu().data.numpy()

    # pool = Pool(processes=30)
    # X = pool.map(get_emb, X)
    # X = np.array(X)
    # print(np.array(X[0][0][0]).shape)

    # savepkl('./data/xp_2D_10-50_emb.pkl', X)

    baseline_f = pd.read_csv('./baseline_f_t-emb.csv')
    # baseline_f['table'] = baseline_f.table_id.parallel_apply(
    #     lambda x: tokenize_table(read_table(x)['data']))
    # baseline_f['table_emb'] = baseline_f.table.parallel_apply(
    #     lambda x: get_emb(x))
    print(baseline_f[baseline_f['table_emb'] == 'nan'])
    baseline_f['table_emb'] = baseline_f.table_emb.apply(eval)
    baseline_f['query_emb'] = baseline_f.query.apply(
        lambda x: tokenize_cell(x))
    print(baseline_f.head())
    # baseline_f = mp(baseline_f, t, 20)
    # print(baseline_f.iloc[:2]['table_emb'])
    # baseline_f.to_csv('./baseline_f_t-emb.csv',index=False)
    # semantic_f['w2v_early_fusion'] = semantic_f.apply(
    #     lambda x: early_fusion(x['w2v_embd_table'], x['w2v_embd_query']), axis=1)

    # semantic_f['w2v_late_fusion'] = semantic_f.parallel_apply(
    #     lambda x: late_fusion(x['w2v_embd_table'], x['w2v_embd_query']), axis=1)
    # semantic_f['w2v_late_fusion_max'] = semantic_f.w2v_late_fusion.parallel_apply(
    #     np.max)
    # semantic_f['w2v_late_fusion_avg'] = semantic_f.w2v_late_fusion.parallel_apply(
    #     np.average)
    # semantic_f['w2v_late_fusion_sum'] = semantic_f.w2v_late_fusion.parallel_apply(
    #     np.sum)
