import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dataset import T2VDataset
from trec import TREC_data_prep, TREC_model, mp
from TREC_score import ndcg_pipeline
from utils import loadpkl, savepkl

logger = logging.getLogger("app")


def fit(model, pos_sample, neg_sample, vocab, loss_fn, opt, config, output_dir, writers, device):
    def trec_eval(model):
        logger.info('TREC model building...')

        baseline_f = pd.read_csv(config['input_files']['baseline_f'])
        trec = TREC_data_prep(model=model, vocab=vocab)
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

    Xp, yp = pos_sample
    Xn, yn = neg_sample
    train_writer, test_writer = writers
    epochs = config['model_params']['epochs']
    model_name = os.path.join(output_dir, config['model_props']['model_name'])

    train_writer.add_graph(model, torch.ones(
        [1, 50, 10, 20], dtype=torch.long, device=device))

    '''
    Creating + and - dataset
    '''
    dataset_p = T2VDataset(Xp, yp, vocab, device, config)
    dataset_n = T2VDataset(Xn, yn, vocab, device, config)

    '''
    Splitting the +ve dataset into train and test sets
    '''
    train_size = int(0.7 * len(dataset_p))
    test_size = len(dataset_p) - train_size
    train_dataset_p, test_dataset_p = random_split(
        dataset_p, [train_size, test_size])

    # logger.info(f"+ve trainset: {len(train_dataset_p)}, +ve testset: {len(test_dataset_p)}")
    start_time_total = time.time()
    ndcg_scores_total = {}

    for epoch in range(epochs):
        start_time_epoch = time.time()
        logger.info(f"Epoch: {epoch+1}/{epochs}")

        '''
        Splitting the -ve dataset into train and test sets every epoch
        '''
        # train_Xn, test_Xn = train_test_split(Xn, train_size=0.7)
        # train_yn, test_yn = train_test_split(yn, train_size=0.7)
        train_size = int(0.7 * len(dataset_n))
        test_size = len(dataset_n) - train_size
        train_dataset_n, test_dataset_n = random_split(
            dataset_n, [train_size, test_size])

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
        train_dataset = ConcatDataset([train_dataset_p, train_dataset_n])
        dataloader = DataLoader(
            train_dataset, batch_size=config['model_params']['batch_size'], shuffle=True)
        # logger.info(f"-ve X trainset: {Xn_.shape}, -ve y trainset: {yn_.shape}, -ve trainset: {len(train_dataset_n)}, total trainset: {len(train_dataset)}")

        for X_batch, y_batch in tqdm(dataloader):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            y_actual_train += list(y_batch.cpu().data.numpy())
            y_p = torch.round(y_pred)
            y_pred_train += list(y_p.cpu().data.numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_per_epoch += loss.item()

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

        '''--------------------------------------------------------------------------------------'''

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
        test_dataset = ConcatDataset([test_dataset_p, test_dataset_n])
        dataloader = DataLoader(
            test_dataset, batch_size=config['model_params']['batch_size'], shuffle=True)
        # logger.info(f"-ve X testset: {Xn_.shape}, -ve y testset: {yn_.shape}, -ve testset: {len(test_dataset_n)}, total testset: {len(test_dataset)}")

        for X_batch, y_batch in tqdm(dataloader):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            y_p = torch.round(y_pred)
            y_actual_test += list(y_batch.cpu().data.numpy())
            y_pred_test += list(y_p.cpu().data.numpy())
            loss_per_epoch += loss.item()

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

        if config['trec']['compute']:
            ndcg_score, ndcg_score_dict = trec_eval(model)
            logger.info(f"\n{ndcg_score}")
            for ndcg_type in ndcg_score_dict.keys():
                train_writer.add_scalar(
                    f'NDCG scores/{ndcg_type}', ndcg_score_dict[ndcg_type], epoch)
                if ndcg_type not in ndcg_scores_total.keys():
                    ndcg_scores_total[ndcg_type] = []
                else:
                    ndcg_scores_total[ndcg_type].append(
                        ndcg_score_dict[ndcg_type])
            # train_writer.add_scalars(f'NDCG scores', ndcg_score_dict, epoch)

        end_time_epoch = time.time()-start_time_epoch
        logger.info(f"Time spent in this epoch : {end_time_epoch}\n")

        train_writer.flush()
        test_writer.flush()

    end_time_total = time.time()-start_time_total
    logger.info(f"Time spent total : {end_time_total}\n")
    torch.save(model.state_dict(), model_name)
    plot_ndcg_epochs(ndcg_scores_total, output_dir, config)


def make_writer(output_dir, writer_type, config):
    path = os.path.join(output_dir, config['model_props']['viz_path'])
    utils.make_dirs(path)
    path = os.path.join(path, writer_type)
    utils.make_dirs(path)
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
