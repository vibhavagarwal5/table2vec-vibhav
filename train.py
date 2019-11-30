import time
import logging
import numpy as np
import pandas as pd
import random
import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import utils
from preprocess import loadpkl
from dataset import T2VDataset
from model import Table2Vec
from trec import TREC_data_prep, TREC_model
from TREC_score import ndcg_pipeline

logger = logging.getLogger("app")


def fit(model, pos_sample, neg_sample, loss_fn, opt, config, output_dir, writer, device):
    Xp, yp = pos_sample
    Xn, yn = neg_sample
    train_writer, test_writer = writer
    epochs = config['model_params']['epochs']
    model_name = os.path.join(output_dir, config['model_props']['model_name'])

    train_writer.add_graph(model, torch.ones(
        [1, 50, 10, 20], dtype=torch.long, device=device))

    # start_p_time = time.time()
    dataset_p = T2VDataset(Xp, yp, vocab, device, config)
    train_size = int(0.7 * len(dataset_p))
    test_size = len(dataset_p) - train_size
    train_dataset_p, test_dataset_p = random_split(
        dataset_p, [train_size, test_size])

    # logger.info(f"+ve trainset: {len(train_dataset_p)}, +ve testset: {len(test_dataset_p)}")

    # end_p_time = time.time()-start_p_time
    # logger.info(f"Time spent in creating the training dataset: {end_p_time}")

    for epoch in range(epochs):
        start_time = time.time()
        logger.info(f"Epoch: {epoch+1}/{epochs}")
        train_Xn, test_Xn = train_test_split(Xn, train_size=0.7)
        train_yn, test_yn = train_test_split(yn, train_size=0.7)

        '''Training'''
        loss_per_epoch = 0
        y_actual_train = list()
        y_pred_train = list()

        # start_train_time = time.time()
        Xn_ = np.array(random.sample(train_Xn.tolist(), len(train_dataset_p)))
        yn_ = np.array(random.sample(train_yn.tolist(), len(train_dataset_p)))
        train_dataset_n = T2VDataset(Xn_, yn_, vocab, device, config)
        train_dataset = ConcatDataset([train_dataset_p, train_dataset_n])
        # logger.info(f"-ve X trainset: {Xn_.shape}, -ve y trainset: {yn_.shape}, -ve trainset: {len(train_dataset_n)}, total trainset: {len(train_dataset)}")

        # end_train_time = time.time()-start_train_time
        # logger.info(f"Time spent in creating the training dataset: {end_train_time}")

        dataloader = DataLoader(
            train_dataset, batch_size=model_params['batch_size'], shuffle=True)

        for X_batch, y_batch in tqdm(dataloader):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            y_p = torch.round(y_pred)
            y_actual_train += list(y_batch.cpu().data.numpy())
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

        '''Testing'''
        loss_per_epoch = 0
        y_actual_test = list()
        y_pred_test = list()

        # start_test_time = time.time()
        Xn_ = np.array(random.sample(test_Xn.tolist(), len(test_dataset_p)))
        yn_ = np.array(random.sample(test_yn.tolist(), len(test_dataset_p)))
        test_dataset_n = T2VDataset(Xn_, yn_, vocab, device, config)
        test_dataset = ConcatDataset([test_dataset_p, test_dataset_n])
        # logger.info(f"-ve X testset: {Xn_.shape}, -ve y testset: {yn_.shape}, -ve testset: {len(test_dataset_n)}, total testset: {len(test_dataset)}")

        # end_test_time = time.time()-start_test_time
        # logger.info(f"Time spent in creating the testing dataset: {end_test_time}")

        dataloader = DataLoader(
            test_dataset, batch_size=model_params['batch_size'], shuffle=True)

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

        end_time = time.time()-start_time
        logger.info(f"TIme spent in this epoch : {end_time}\n")

        train_writer.flush()
        test_writer.flush()
    torch.save(model.state_dict(), model_name)


def make_writer(output_dir, writer_type, config):
    path = os.path.join(output_dir, config['model_props']['viz_path'])
    utils.make_dirs(path)
    path = os.path.join(path, writer_type)
    utils.make_dirs(path)
    return SummaryWriter(log_dir=path, comment=config['comment'])


if __name__ == "__main__":
    output_dir, config = utils.setup_simulation()
    model_params = config['model_params']
    input_files = config['input_files']
    trec_config = config['trec']

    torch.manual_seed(model_params['seed'])

    Xp = loadpkl(input_files['Xp_path'])
    Xn = loadpkl(input_files['Xn_path'])
    yp = loadpkl(input_files['yp_path'])
    yn = loadpkl(input_files['yn_path'])
    vocab = loadpkl(input_files['vocab_path'])
    logger.info(f"Xp.shape: {Xp.shape}, yp.shape: {yp.shape}")
    logger.info(f"Xn.shape: {Xn.shape}, yn.shape: {yn.shape}")
    logger.info(f"len(vocab): {len(vocab)}")

    device = torch.device(
        f"cuda:{config['CUDA_NO']}" if torch.cuda.is_available() else 'cpu')

    train_writer = make_writer(output_dir, 'train', config)
    test_writer = make_writer(output_dir, 'test', config)

    loss_function = nn.BCELoss()
    model = Table2Vec(len(vocab), model_params['embedding_dim'], device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    fit(model=model, pos_sample=(Xp, yp), neg_sample=(Xn, yn), loss_fn=loss_function,
        opt=optimizer, config=config, output_dir=output_dir, writer=(train_writer, test_writer), device=device)

    train_writer.close()
    test_writer.close()

    logger.info('TREC model building...')
    model_load = torch.load(os.path.join(output_dir, 'model.pt'))
    # model_load = torch.load('./output/11_25_15_56_30/model.pt')
    baseline_f = pd.read_csv(input_files['baseline_f'])

    trec = TREC_data_prep(model=model_load, vocab=vocab)
    baseline_f = trec.mp(df=baseline_f, func=trec.pipeline, num_partitions=20)
    baseline_f.drop(columns=['table_emb', 'query_emb'], inplace=True)
    # baseline_f.to_csv('./baseline_f_tq-emb_temp.csv', index=False)
    # baseline_f = pd.read_csv('./baseline_f_tq-emb_temp.csv')

    logger.info('TREC NDCG scoring....')
    trec_path = os.path.join(output_dir, trec_config['folder_name'])
    trec_model = TREC_model(data=baseline_f, output_dir=trec_path, config=config)
    trec_model.train()

    ndcg_pipeline(trec_model.file_path, trec_config['trec_path'], trec_config['query_file_path'])
