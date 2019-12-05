import argparse
from pandarallel import pandarallel
import numpy as np
import pandas as pd
import os
import torch
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
import logging
import time

from preprocess import loadpkl, tokenize_table, read_table, tokenize_cell
import utils
from utils import Config

# pandarallel.initialize(progress_bar=True, nb_workers=20, shm_size_mb=3000)
logger = logging.getLogger("app")


class TREC_data_prep():

    def __init__(self, model, vocab):
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.embeddings = model['embeddings.weight'].cpu().data.numpy()

    def get_emb(self, table):
        if len(table) == 0:
            return [self.embeddings[self.w2i['<PAD>']]]
        for row in table:
            for j, cell in enumerate(row):
                if len(row[j]) == 0:
                    row[j].append('<PAD>')
                for i, item in enumerate(cell):
                    cell[i] = self.embeddings[self.w2i[item]]
                row[j] = np.average(row[j], axis=0).tolist()
        x = np.array(table)
        shape = x.shape
        table = x.reshape(shape[0]*shape[1], shape[2])
        return table.tolist()

    def late_fusion(self, table, query):
        s = []
        for i in query:
            for j in table:
                i = np.array(i)
                j = np.array(j)
                sim = cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))
                s.append(sim)
        s = np.array(s).reshape(-1)
        return s

    def early_fusion(self, table, query):
        a = np.average(table, axis=0).reshape(1, -1)
        b = np.average(query, axis=0).reshape(1, -1)
        sim = cosine_similarity(a, b)
        return sim.reshape(-1)[0]

    def pipeline(self, baseline_f):
        baseline_f['table_emb'] = baseline_f.table_tkn.apply(
            lambda x: self.get_emb(eval(x)))
        baseline_f['query_emb'] = baseline_f.query_tkn.apply(
            lambda x: [self.embeddings[self.w2i[item]] for item in eval(x)])

        baseline_f['early_fusion'] = baseline_f.apply(
            lambda x: self.early_fusion(x['table_emb'], x['query_emb']), axis=1)
        baseline_f['late_fusion'] = baseline_f.apply(
            lambda x: self.late_fusion(x['table_emb'], x['query_emb']), axis=1)

        baseline_f['late_fusion_max'] = baseline_f.late_fusion.apply(
            np.max)
        baseline_f['late_fusion_avg'] = baseline_f.late_fusion.apply(
            np.average)
        baseline_f['late_fusion_sum'] = baseline_f.late_fusion.apply(
            np.sum)
        return baseline_f


class TREC_model():
    def __init__(self, data, output_dir, config):
        self.data = data
        self.config = config
        self.file_path = os.path.join(output_dir, config['trec']['file_name'])
        self.prep_data()
        utils.make_dirs(output_dir)

    def prep_data(self):
        x_bf = ['row', 'col', 'nul', 'in_link', 'out_link', 'pgcount', 'tImp', 'tPF', 'leftColhits', 'SecColhits', 'bodyhits', 'PMI', 'qInPgTitle', 'qInTableTitle', 'yRank', 'csr_score', 'idf1',
                'idf2', 'idf3', 'idf4', 'idf5', 'idf6', 'max', 'sum', 'avg', 'sim', 'emax', 'esum', 'eavg', 'esim', 'cmax', 'csum', 'cavg', 'csim', 'remax', 'resum', 'reavg', 'resim', 'query_l']
        x_smf = ['early_fusion', 'late_fusion_max',
                 'late_fusion_avg', 'late_fusion_sum']
        x_f = x_bf
        y_f = ['rel']
        if self.config['trec']['semantic_f']:
            x_f += x_smf

        self.X = self.data[x_f]
        self.y = self.data[y_f]

    def train(self):
        kfold = KFold(5, True, 42)

        for i, indices in enumerate(kfold.split(self.X)):
            train_idx, test_idx = indices
            X_train, X_test, y_train, y_test = self.X.iloc[train_idx], self.X.iloc[
                test_idx], self.y.iloc[train_idx], self.y.iloc[test_idx]
            df = self.makeModel_getdf(X_train, X_test, y_train, y_test)
            df.to_csv(f"{self.file_path}{i}.txt",
                      sep=' ', index=False, header=False)

    def makeModel_getdf(self, X_train, X_test, y_train, y_test):
        self.clf = RandomForestClassifier(
            n_estimators=1000, max_features=3, random_state=42)
        self.clf.fit(X_train, y_train.values.ravel())
        X_test = mp(X_test, self.score_mp, 20)
        df = self.generate_trec_df(self.generate_filtered_df(X_test, y_test))
        return df

    def score_mp(self, X_test):
        X_test['model_score'] = X_test.apply(
            lambda x: self.getScore(x), axis=1)
        return X_test

    def getScore(self, row):
        arr = self.clf.predict_proba(np.array(row).reshape(1, -1))
        return arr[0][1]+2*arr[0][2]

    def generate_filtered_df(self, X, y):
        df = pd.concat([
            self.data.iloc[list(X.index)][['query_id', 'query', 'table_id']],
            X['model_score']], axis=1)
        return df

    def generate_trec_df(self, df):
        l = []
        dic = dict(df.query_id.value_counts())
        for i in dic:
            for j in range(1, dic[i]+1):
                l.append(j)

        df_temp = pd.DataFrame()
        df_temp['query_id'] = df['query_id']
        df_temp['Q0'] = 'Q0'
        df_temp['table_id'] = df['table_id']
        df_temp['rank'] = l
        df_temp['score'] = df['model_score']
        df_temp['smarttable'] = 'smarttable'
        return df_temp


def mp(df, func, num_partitions):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_partitions)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="path for the scores")
    parser.add_argument("--data_prep",
                        help="path for the scores", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    start_time = time.time()

    if args.data_prep:
        print('Data preparation happening...')
        baseline_f = pd.read_csv('../global_data/features.csv')

        def t(baseline_f):
            baseline_f['table_tkn'] = baseline_f.table_id.apply(
                lambda x: tokenize_table(read_table(x)['data']))
            baseline_f['query_tkn'] = baseline_f['query'].apply(
                lambda x: tokenize_cell(x))
            return baseline_f
        baseline_f = mp(baseline_f, t, 20)
        baseline_f.to_csv('./baseline_f_tq-tkn.csv', index=False)

    vocab = loadpkl('./data/vocab_2D_10-50_complete.pkl')
    output_dir = f'./output/{args.path}'

    config = Config()
    model_load = torch.load(os.path.join(output_dir, 'model.pt'))
    # model_load = torch.load('./output/11_25_15_56_30/model.pt')
    baseline_f = pd.read_csv('./baseline_f_tq-tkn.csv')

    trec = TREC_data_prep(model=model_load, vocab=vocab)
    baseline_f = mp(
        df=baseline_f, func=trec.pipeline, num_partitions=20)
    baseline_f.drop(columns=['table_emb', 'query_emb'], inplace=True)
    # baseline_f.to_csv('./baseline_f_tq-emb_temp.csv', index=False)
    # baseline_f = pd.read_csv('./baseline_f_tq-emb_temp.csv')

    trec_path = os.path.join(output_dir, config['trec']['folder_name'])
    trec_model = TREC_model(
        data=baseline_f, output_dir=trec_path, config=config)
    trec_model.train()
    print(time.time() - start_time)
