import argparse
from collections import OrderedDict
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from data_preprocess2 import split_overflow_table
from utils import Config, loadpkl, make_dirs, mp, flatten_1_deg
import models
from dataset import T2VDataset

logger = logging.getLogger("app")


class TREC_data_prep():
    def __init__(self, model, config, vocab):
        self.model = model
        self.config = config
        self.vocab = vocab

    def convert2table(self, inp, typ):
        if typ == 'table':
            if len(inp) == 0:
                inp = [[['<PAD>']]]
            else:
                for row in inp:
                    for j, cell in enumerate(row):
                        if len(row[j]) == 0:
                            row[j].append('<PAD>')
        if typ == 'query':
            inp = [[[j] for j in inp]]
        return inp

    def prepare_data(self, inp, typ):
        inp = split_overflow_table(inp)

        for i in range(len(inp)):
            # print(np.array(inp[i]).shape, inp[i])
            inp[i] = T2VDataset.pad_table(
                self.config['table_prep_params'], inp[i], '<PAD>')
        inp = T2VDataset.table_words2index(self.vocab, inp)
        return np.array(inp)

    def duplicate_rows(self, df, typ):
        for index, row in df.iterrows():
            rows2add = row * row[typ].shape[0]
            print(rows2add)
            for i in range(row[typ].shape[0]):
                rows2add[i][typ] = rows2add[i][typ][i]
            df = df.drop(index)
            df = pd.concat([df, rows2add])
        # rows2add = []
        # temp = row[typ]
        # for i in temp:
        #     row[tmp] = i
        #     rows2add.append(row)
        return df

    def late_fusion(self, table, query):
        s = []
        for i in query:
            for j in table:
                sim = cosine_similarity(
                    np.array(i).reshape(1, -1),
                    np.array(j).reshape(1, -1))
                s.append(sim)
        s = np.array(s).reshape(-1)
        return s

    def early_fusion(self, table, query):
        a = np.average(table, axis=0).reshape(1, -1)
        b = np.average(query, axis=0).reshape(1, -1)
        sim = cosine_similarity(a, b)
        return sim.reshape(-1)[0]

    def pipeline(self, baseline_f):
        baseline_f['table_1'] = baseline_f.table_tkn.apply(
            lambda x: self.convert2table(eval(x), 'table'))
        baseline_f['query_1'] = baseline_f.query_tkn.apply(
            lambda x: self.convert2table(eval(x), 'query'))
        baseline_f['table_ft'] = baseline_f['table_1'].apply(
            lambda x: self.prepare_data(x, 'table'))
        baseline_f['query_ft'] = baseline_f['query_1'].apply(
            lambda x: self.prepare_data(x, 'query'))
        print(baseline_f['table_ft'].apply(lambda x: x.shape))
        baseline_f = self.duplicate_rows(baseline_f, 'table_ft')
        print(baseline_f['table_ft'].apply(lambda x: x.shape))

        # baseline_f['early_fusion'] = baseline_f.apply(
        #     lambda x: self.early_fusion(x['table_ft'], x['query_ft']), axis=1)
        # baseline_f['late_fusion'] = baseline_f.apply(
        #     lambda x: self.late_fusion(x['table_ft'], x['query_ft']), axis=1)

        # baseline_f['late_fusion_max'] = baseline_f.late_fusion.apply(
        #     np.max)
        # baseline_f['late_fusion_avg'] = baseline_f.late_fusion.apply(
        #     np.average)
        # baseline_f['late_fusion_sum'] = baseline_f.late_fusion.apply(
        #     np.sum)
        return baseline_f


class TREC_model():
    def __init__(self, data, output_dir, config):
        self.data = data
        self.config = config
        self.file_path = os.path.join(output_dir, config['trec']['file_name'])
        self.prep_data()
        make_dirs(output_dir)

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
        # self.clf = XGBClassifier(
            # tree_method='gpu_hist',
            # gpu_id=self.config['gpu']
            # )
        # self.clf = AdaBoostClassifier(
        #     n_estimators=1000,
        #     learning_rate=1,
        #     random_state=42)
        self.clf = RandomForestClassifier(
            n_estimators=1000,
            max_features=3,
            random_state=42)
        self.clf.fit(X_train, y_train.values.ravel())
        # self.clf.fit(X_train.values, y_train.values)
        # X_test = self.score_mp(X_test)
        X_test = mp(X_test, self.score_mp, 20)
        df = self.generate_trec_df(self.generate_filtered_df(X_test, y_test))
        return df

    def score_mp(self, X_test):
        X_test['model_score'] = X_test.apply(
            lambda x: self.getScore(x), axis=1)
        return X_test

    def getScore(self, row):
        arr = self.clf.predict_proba(np.array(row).reshape(1, -1))
        return arr[0][1] + 2 * arr[0][2]

    def generate_filtered_df(self, X, y):
        df = pd.concat([
            self.data.iloc[list(X.index)][['query_id', 'query', 'table_id']],
            X['model_score']], axis=1)
        return df

    def generate_trec_df(self, df):
        l = []
        dic = dict(df.query_id.value_counts())
        for i in dic:
            for j in range(1, dic[i] + 1):
                l.append(j)

        df_temp = pd.DataFrame()
        df_temp['query_id'] = df['query_id']
        df_temp['Q0'] = 'Q0'
        df_temp['table_id'] = df['table_id']
        df_temp['rank'] = l
        df_temp['score'] = df['model_score']
        df_temp['smarttable'] = 'smarttable'
        return df_temp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="path for the scores")
    parser.add_argument("-m", "--model_name")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    start_time = time.time()

    config = Config()
    config.load(os.path.join(args.path, 'config.toml'))

    vocab = loadpkl(config['input_files']['vocab_path'])
    device = torch.device(f"cuda:{config['gpu']}")

    # model = models.create_model(
    #     config['model_props']['type'],
    #     params=(
    #         len(vocab),
    #         config['model_params']['embedding_dim'],
    #         device
    #     )
    # )
    # model.to(device)
    # state_dict = torch.load(os.path.join(args.path, args.model_name))
    # model.linear_layers = torch.nn.Sequential(
    #     *(list(model.linear_layers.children())[:2]))
    # state_dict_ = OrderedDict(
    #     {i: state_dict[i] for i in list(model.state_dict())})
    # model.load_state_dict(state_dict_)
    # print(torch.equal(list(model.parameters())[
    #       0], state_dict_['embeddings.weight']))
    # print(model)
    # model.eval()
    model = None
    baseline_f = pd.read_csv(config['input_files']['baseline_f'])
    trec = TREC_data_prep(model, config, vocab)
    baseline_f = trec.pipeline(baseline_f.iloc[:10, :])
    # baseline_f = mp(
    #     df=baseline_f, func=trec.pipeline, num_partitions=20)

    # if args.data_prep and args.data_path:
    #     print('Data preparation happening...')
    #     baseline_f = pd.read_csv('../global_data/features.csv')

    #     def t(baseline_f):
    #         baseline_f['table_tkn'] = baseline_f.table_id.apply(
    #             lambda x: shrink_cell_len(clean(tokenize_table(read_table(x)['data']))))
    #         baseline_f['query_tkn'] = baseline_f['query'].apply(
    #             lambda x: tokenize_str(x.lower()))
    #         return baseline_f

    #     baseline_f = mp(baseline_f, t, 50)
    #     baseline_f.to_csv(os.path.join(
    #         args.data_path, 'baseline_f_tq-tkn.csv'), index=False)

    # if args.path:
    #     config = Config()
    #     vocab = loadpkl(config['input_files']['vocab_path'])
    #     output_dir = f'./output/{args.path}'
    #     model_load = torch.load(os.path.join(output_dir, 'model.pt'))
    #     baseline_f = pd.read_csv(config['input_files']['baseline_f'])

    #     trec = TREC_data_prep(embeddings=model_load.embeddings, vocab=vocab)
    #     baseline_f = mp(
    #         df=baseline_f, func=trec.pipeline, num_partitions=20)
    #     baseline_f.drop(columns=['table_emb', 'query_emb'], inplace=True)
    #     # baseline_f.to_csv('./baseline_f_tq-emb_temp.csv', index=False)
    #     # baseline_f = pd.read_csv('./baseline_f_tq-emb_temp.csv')

    #     trec_path = os.path.join(output_dir, config['trec']['folder_name'])
    #     trec_model = TREC_model(
    #         data=baseline_f, output_dir=trec_path, config=config)
    #     trec_model.train()
    # print(time.time() - start_time)
