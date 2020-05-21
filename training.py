import logging
import os
import random
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
from dataset import T2VDataset
from trec import TREC_data_prep, TREC_model
from TREC_score import ndcg_pipeline
from utils import flatten_1_deg, loadpkl, make_dirs, mp, savepkl

logger = logging.getLogger("app")
prev_embd = None


def print_table(table, vocab):
    i2w = {i: w for i, w in enumerate(vocab)}
    table = table.tolist()
    for r in table:
        for c in r:
            for i, w in enumerate(c):
                c[i] = i2w[int(w)]
        logger.info(r)


# def fit(model, pos_sample, neg_sample, vocab, loss_fn, opt, config, output_dir, writers, device):
def fit(pos_sample, neg_sample, vocab, config, output_dir):
    global prev_embd

    def trec_eval(embeddings):
        logger.info('TREC model building...')
        baseline_f = pd.read_csv(config['input_files']['baseline_f'])
        trec = TREC_data_prep(embeddings=embeddings, vocab=vocab)
        baseline_f = mp(
            df=baseline_f, func=trec.pipeline, num_partitions=20)
        baseline_f.drop(columns=['table_emb', 'query_emb'], inplace=True)

        logger.info('TREC NDCG scoring....')
        trec_path = os.path.join(output_dir, config['trec']['folder_name'])
        trec_model = TREC_model(
            data=baseline_f, output_dir=trec_path, config=config)
        trec_model.train()

        ndcg_score, ndcg_score_dict = ndcg_pipeline(trec_model.file_path,
                                                    config['trec']['trec_path'], config['trec']['query_file_path'])
        return ndcg_score, ndcg_score_dict

    def generate_rand_cell(table, table_lst):
        def get_rand_table():
            rand_table_name = random.choice(range(len(all_tables)))
            t_d = all_tables[rand_table_name].tolist()
            return t_d, rand_table_name

        table_data, table_name = get_rand_table()
        numDataRows, numCols = np.array(table_data).shape[:2]
        rand_row_ix = random.choice(list(range(numDataRows)))
        rand_col_ix = random.choice(list(range(numCols)))
        rand_cell = table_data[rand_row_ix][rand_col_ix]

        # while (
        #     # True
        #     # rand_cell in flatten_1_deg(table) or
        #     # table_name in table_lst or
        #     rand_cell.count(1) == len(rand_cell)
        # ):
        #     # if table_name in table_lst:
        #     #     return generate_rand_cell(table, table_lst)
        #     # elif rand_cell.count(1) == len(rand_cell):
        #     #     rand_row_ix = random.choice(list(range(numDataRows)))
        #     #     rand_col_ix = random.choice(list(range(numCols)))
        #     #     rand_cell = table_data[rand_row_ix][rand_col_ix]
        #     # else:
        #     #     break
        #     # print('sample table again coz repeat')
        #     rand_cell, table_name = generate_rand_cell(table, table_lst)
        return rand_cell, table_name

    def generate_neg_table(inp):
        t_l = []
        t = []
        for i in range(config['table_prep_params']['MAX_ROW_LEN']):
            r = []
            for j in range(config['table_prep_params']['MAX_COL_LEN']):
                c, t_name = generate_rand_cell(t, t_l)
                r.append(c)
                t_l.append(t_name)
            t.append(r)
        return t

    def generate_neg(size):
        Xn = [generate_neg_table(i) for i in range(size)]
        # p = Pool(processes=20)
        # Xn = p.map(generate_neg_table, range(size))
        # p.close()
        # p.join()

        yn = [0] * size
        Xn = torch.tensor(Xn, device=device)
        yn = torch.tensor(yn, dtype=torch.float, device=device)
        return Xn, yn.reshape((-1, 1))

    Xp, yp = pos_sample
    # Xn, yn = neg_sample
    args, config = config
    train_writer = make_writer(output_dir, 'train', config)
    test_writer = make_writer(output_dir, 'test', config)
    epochs = config['model_params']['epochs']
    model_name = os.path.join(output_dir, config['model_props']['model_name'])

    device = torch.device(f"cuda:{config['gpu']}")
    model = models.create_model(
        config['model_props']['type'],
        params=(
            len(vocab),
            config['model_params']['embedding_dim'],
            device
        )
    )
    if args.use_checkpoint is not None:
        model.load_state_dict(torch.load(args.use_checkpoint))
        logger.info('Using checkpoint from : {}'.format(args.use_checkpoint))
    else:
        logger.info('Using fresh model')
    model = model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=config['model_params']['opt_lr'])

    # train_writer.add_graph(model, torch.ones(
    #     [1, 50, 10, 20], dtype=torch.long, device=device))

    '''
    Creating + and - dataset
    '''
    dataset_p = T2VDataset(Xp, yp, vocab, device, config)
    # dataset_n = T2VDataset(Xn, yn, vocab, device, config)

    '''
    Splitting the +ve dataset into train and test sets
    '''
    train_size = int(0.7 * len(dataset_p))
    test_size = len(dataset_p) - train_size
    train_dataset_p, test_dataset_p = random_split(
        dataset_p, [train_size, test_size])

    start_time_total = time.time()
    ndcg_scores_total = {}

    for epoch in range(epochs):
        start_time_epoch = time.time()
        logger.info(f"Epoch: {epoch+1}/{epochs}")

        '''
        Splitting the -ve dataset into train and test sets every epoch
        '''
        # # p = Pool(processes=5)
        # # Xn = p.map(generate_neg_table, range(int(len(Xp)*1.2)))
        # # p.close()
        # # p.join()
        # Xn = [generate_neg_table(i) for i in range(int(len(Xp)*1.2))]
        # Xn = np.array(Xn)
        # print(Xn.shape)
        # dataset_n = T2VDataset(Xn, -1, vocab, device, config)

        # # train_Xn, test_Xn = train_test_split(Xn, train_size=0.7)
        # # train_yn, test_yn = train_test_split(yn, train_size=0.7)
        # train_size = int(0.7 * len(dataset_n))
        # test_size = len(dataset_n) - train_size
        # train_dataset_n, test_dataset_n = random_split(
        #     dataset_n, [train_size, test_size])

        '''
        Training
        '''
        loss_per_epoch = 0
        y_actual_train = list()
        y_pred_train = list()

        '''
        Creating training dataset
        '''
        # Xn_ = np.array(random.sample(train_Xn.tolist(), len(train_dataset_p)))
        # yn_ = np.array(random.sample(train_yn.tolist(), len(train_dataset_p)))
        # train_dataset_n = T2VDataset(Xn_, yn_, vocab, device, config)
        # train_dataset = ConcatDataset([train_dataset_p, train_dataset_n])
        dataloader = DataLoader(
            train_dataset_p,
            batch_size=config['model_params']['batch_size'],
            shuffle=True,
        )

        for index, (X_batch, y_batch) in enumerate(tqdm(dataloader)):
            all_tables = X_batch
            Xn, yn = generate_neg(len(X_batch))
            total_inputs = torch.cat((X_batch, Xn), dim=0)
            y_batch_total = torch.cat((y_batch, yn), dim=0)
            shuffle = torch.randperm(len(total_inputs))
            total_inputs = total_inputs[shuffle]
            y_batch_total = y_batch_total[shuffle]

            y_pred = model(total_inputs)
            loss = loss_fn(y_pred, y_batch_total)
            y_actual_train += list(y_batch_total.cpu().data.numpy())
            y_p = torch.round(y_pred)
            y_pred_train += list(y_p.cpu().data.numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_per_epoch += loss.item()

            if index % config['model_props']['print_every'] == 0:
                logger.info(f'TRAIN: {index}')
                logger.info(
                    f"INPUT TABLE:\n{print_table(total_inputs[0].cpu().data.numpy(),vocab)}")
                logger.info(
                    f'GOLD LABEL: {y_batch_total[0].cpu().data.numpy()}')
                logger.info(f'PREDICTED LABEL: {y_p[0].cpu().data.numpy()}\n')

        accuracy = accuracy_score(y_actual_train, y_pred_train)
        precision = precision_score(y_actual_train, y_pred_train)
        recall = recall_score(y_actual_train, y_pred_train)
        f1 = f1_score(y_actual_train, y_pred_train)

        train_writer.add_scalar('Loss', loss_per_epoch, epoch)
        train_writer.add_scalar('Accuracy', accuracy, epoch)
        train_writer.add_scalar('F1', f1, epoch)
        train_writer.add_scalar('Precision', precision, epoch)
        train_writer.add_scalar('Recall', recall, epoch)
        logger.info(
            f"Training - Loss : {loss_per_epoch}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}")

        '''
        --------------------------------------------------------------------------------------
        '''

        '''
        Testing
        '''
        loss_per_epoch = 0
        y_actual_test = list()
        y_pred_test = list()

        '''
        Creating testing dataset
        '''
        # Xn_ = np.array(random.sample(test_Xn.tolist(), len(test_dataset_p)))
        # yn_ = np.array(random.sample(test_yn.tolist(), len(test_dataset_p)))
        # test_dataset_n = T2VDataset(Xn_, yn_, vocab, device, config)
        # test_dataset = ConcatDataset([test_dataset_p, test_dataset_n])
        dataloader = DataLoader(
            test_dataset_p,
            batch_size=config['model_params']['batch_size'],
            shuffle=True,
        )

        for X_batch, y_batch in tqdm(dataloader):

            all_tables = X_batch
            Xn, yn = generate_neg(len(X_batch))
            total_inputs = torch.cat((X_batch, Xn), dim=0)
            y_batch_total = torch.cat((y_batch, yn), dim=0)
            shuffle = torch.randperm(len(total_inputs))
            total_inputs = total_inputs[shuffle]
            y_batch_total = y_batch_total[shuffle]

            y_pred = model(total_inputs)
            loss = loss_fn(y_pred, y_batch_total)
            y_actual_test += list(y_batch_total.cpu().data.numpy())
            y_p = torch.round(y_pred)
            y_pred_test += list(y_p.cpu().data.numpy())
            loss_per_epoch += loss.item()

            if index % config['model_props']['print_every'] == 0:
                logger.info(f'TEST: {index}')
                logger.info(
                    f"INPUT TABLE:\n{print_table(total_inputs[0].cpu().data.numpy(),vocab)}")
                logger.info(
                    f'GOLD LABEL: {y_batch_total[0].cpu().data.numpy()}')
                logger.info(f'PREDICTED LABEL: {y_p[0].cpu().data.numpy()}\n')

        accuracy = accuracy_score(y_actual_test, y_pred_test)
        precision = precision_score(y_actual_test, y_pred_test)
        recall = recall_score(y_actual_test, y_pred_test)
        f1 = f1_score(y_actual_test, y_pred_test)

        test_writer.add_scalar('Loss', loss_per_epoch, epoch)
        test_writer.add_scalar('Accuracy', accuracy, epoch)
        test_writer.add_scalar('F1', f1, epoch)
        test_writer.add_scalar('Precision', precision, epoch)
        test_writer.add_scalar('Recall', recall, epoch)
        logger.info(
            f"Testing - Loss : {loss_per_epoch}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}")

        embeddings = model.embeddings
        if prev_embd is None:
            prev_embd = embeddings.weight.cpu()
        else:
            curr_embd = embeddings.weight.cpu()
            logger.info(
                f"Are the embeddings same as the previous one? -> {torch.equal(prev_embd,curr_embd)}")
            prev_embd = curr_embd

        if config['trec']['compute']:
            ndcg_score, ndcg_score_dict = trec_eval(embeddings)
            logger.info(f"\n{ndcg_score}")
            for ndcg_type in ndcg_score_dict.keys():
                train_writer.add_scalar(
                    f'NDCG scores/{ndcg_type}', ndcg_score_dict[ndcg_type], epoch)
                if ndcg_type not in ndcg_scores_total.keys():
                    ndcg_scores_total[ndcg_type] = []
                else:
                    ndcg_scores_total[ndcg_type].append(
                        ndcg_score_dict[ndcg_type])

        end_time_epoch = time.time() - start_time_epoch
        logger.info(f"Time spent in this epoch : {end_time_epoch}\n")

        train_writer.flush()
        test_writer.flush()
        torch.save(model.state_dict(), os.path.join(
            output_dir, f"model_{epoch}.pt"))

    end_time_total = time.time() - start_time_total
    logger.info(f"Time spent total : {end_time_total}\n")
    torch.save(model.state_dict(), model_name)
    # plot_ndcg_epochs(ndcg_scores_total, output_dir, config)

    train_writer.close()
    test_writer.close()


def make_writer(output_dir, writer_type, config):
    path = os.path.join(output_dir, config['model_props']['viz_path'])
    make_dirs(path)
    path = os.path.join(path, writer_type)
    make_dirs(path)
    return SummaryWriter(log_dir=path, comment=config['comment'])


def plot_ndcg_epochs(ndcg_scores_total, output_dir, config):
    f, axes = plt.subplots(4, 1, figsize=(10, 20))
    baselines = [0.5951, 0.6293, 0.6590, 0.6825]
    path = os.path.join(output_dir, config['model_props']['viz_path'])
    for a, n, b in zip(axes, list(ndcg_scores_total.keys()), baselines):
        a.plot(ndcg_scores_total[n], label=n, color='red')
        a.axhline(y=b, label='baseline')
        a.set_title(n)
        a.legend(loc='center')
    plt.savefig(os.path.join(path, 'ndcg_compare.png'))
