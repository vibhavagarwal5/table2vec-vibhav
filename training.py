import argparse
import copy
import logging
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
from dataset import T2VDataset
from trec import TREC_data_prep, TREC_model
from TREC_score import ndcg_pipeline
from utils import (flatten_1_deg, loadpkl, make_dirs, mp, savepkl,
                   setup_simulation)

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


def make_writer(output_dir, writer_type, config):
    path = os.path.join(output_dir, config['model_props']['viz_path'])
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


class NegSample():
    def __init__(self, all_tables, config):
        self.all_tables = all_tables
        self.config = config
        self.table_prep_params = config['table_prep_params']

    def generate_rand_cell(self, table, table_lst):
        def get_rand_table():
            rand_table_name = random.choice(range(len(self.all_tables)))
            t_d = self.all_tables[rand_table_name]
            return t_d, rand_table_name

        table_data, table_name = get_rand_table()
        numDataRows, numCols = np.array(table_data).shape[:2]
        rand_row_ix = random.choice(list(range(numDataRows)))
        rand_col_ix = random.choice(list(range(numCols)))
        rand_cell = table_data[rand_row_ix][rand_col_ix]

        if rand_cell == '<UNK>' and random.random() < 0.6:
            rand_cell, table_name = self.generate_rand_cell(table, table_lst)

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

    def generate_neg_table(self, inp):
        t_l = []
        t = []
        row_sh, col_sh = np.array(inp).shape[:2]
        self.table_prep_params = {
            'MAX_ROW_LEN': row_sh,
            'MAX_COL_LEN': col_sh,
        }
        for i in range(self.table_prep_params['MAX_ROW_LEN']):
            r = []
            for j in range(self.table_prep_params['MAX_COL_LEN']):
                c, t_name = self.generate_rand_cell(t, t_l)
                r.append(c)
                t_l.append(t_name)
            t.append(r)
        return t

    def generate_neg(self, vocab, device):
        size = len(self.all_tables)
        Xn = [self.generate_neg_table(table) for table in self.all_tables]
        yn = np.ones((size, 1)) * 0
        Xn, yn = T2VDataset(np.array(copy.deepcopy(Xn)), yn, vocab, device,
                            self.config, run_pipe=True).return_all()
        return Xn, yn


def fit(config, output_dir):
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

    global prev_embd
    args, config = config

    # Data files load
    input_files = config['input_files']
    Xp = loadpkl(input_files['Xp_path'])
    yp = np.ones((len(Xp), 1))
    Xp_unpad = loadpkl(input_files['Xp_unpad_path'])
    if args.smaller_data is not None:
        Xp = Xp[:args.smaller_data]
        yp = yp[:args.smaller_data]
        Xp_unpad = Xp_unpad[:args.smaller_data]
    logger.info(f"Xp.shape: {Xp.shape}, yp.shape: {yp.shape}")
    vocab = loadpkl(input_files['vocab_path'])
    logger.info(f"len(vocab): {len(vocab)}")
    logger.info(f"len(Xp_unpad): {len(Xp_unpad)}")

    # TB_writer, distributed, device etc
    train_writer = make_writer(output_dir, 'train', config)
    test_writer = make_writer(output_dir, 'test', config)
    ndcg_scores_total = {}
    epochs = config['model_params']['epochs']
    batch_size = config['model_params']['batch_size']
    # DistributedDataParallel
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
        batch_size = int(batch_size / torch.cuda.device_count())
    else:
        device = torch.device(f"cuda:{args.gpu}")
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    # Model, loss, opt creation
    model = models.create_model(
        config['model_props']['type'],
        params=(
            len(vocab),
            config['model_params']['embedding_dim'],
            device
        )
    )
    # Use checkpoint load if specified
    if args.use_checkpoint is not None:
        model.load_state_dict(torch.load(args.use_checkpoint))
        logger.info('Using checkpoint from : {}'.format(args.use_checkpoint))
    else:
        logger.info('Using fresh model')
    model = model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=config['model_params']['opt_lr'])
    # DistributedDataParallel
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Creating +ve dataset
    dataset_p = T2VDataset(Xp, yp, vocab, device, config)

    # Splitting the +ve dataset into train and test sets
    train_size = int(0.7 * len(dataset_p))
    test_size = len(dataset_p) - train_size
    train_dataset_p, test_dataset_p = random_split(
        dataset_p, [train_size, test_size])
    # X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    #     Xp, yp, range(len(Xp)), train_size=0.7, random_state=config['model_params']['seed'])
    # X_train_unpad, X_test_unpad = Xp_unpad[idx_train], Xp_unpad[idx_test]

    start_time_total = time.time()
    for epoch in range(epochs):
        start_time_epoch = time.time()

        '''
        Training
        '''
        loss_per_epoch = 0
        y_actual_train = list()
        y_pred_train = list()

        # Creating training dataloader
        sampler = None
        if args.distributed:
            sampler = DistributedSampler(train_dataset_p)

        dataloader = DataLoader(train_dataset_p,
                                batch_size=batch_size,
                                shuffle=(sampler is None),
                                sampler=sampler)

        for index, (X_batch, y_batch, idx) in enumerate(tqdm(dataloader)):
            X_batch_unpad = Xp_unpad[idx]
            Xn, yn = NegSample(
                X_batch_unpad, config).generate_neg(vocab, device)
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
                logger.info(f"Epoch: {epoch+1}/{epochs}")
                logger.info(f'TRAIN: {index}')
                logger.info(
                    f"INPUT TABLE:\n{print_table(total_inputs[0].cpu().data.numpy(),vocab)}")
                logger.info(
                    f'GOLD LABEL: {y_batch_total[0].cpu().data.numpy()}')
                logger.info(f'PREDICTED LABEL: {y_p[0].cpu().data.numpy()}\n')
                step = epoch * (len(train_dataset_p) / batch_size) + index
                accuracy_batch = accuracy_score(y_batch_total.cpu().data.numpy(),
                                                y_p.cpu().data.numpy())
                train_writer.add_scalar('Batch_lvl/Loss', loss.item(), step)
                train_writer.add_scalar(
                    'Batch_lvl/Accuracy', accuracy_batch, step)

        accuracy = accuracy_score(y_actual_train, y_pred_train)
        train_writer.add_scalar('Loss', loss_per_epoch, epoch)
        train_writer.add_scalar('Accuracy', accuracy, epoch)
        logger.info(
            f"Training - Epoch: {epoch+1}, Loss: {loss_per_epoch}, Accuracy: {accuracy}")

        '''
        --------------------------------------------------------------------------------------
        '''

        '''
        Testing
        '''
        loss_per_epoch = 0
        y_actual_test = list()
        y_pred_test = list()

        # Creating testing dataloader
        sampler = None
        if args.distributed:
            sampler = DistributedSampler(test_dataset_p)

        dataloader = DataLoader(test_dataset_p,
                                batch_size=batch_size,
                                shuffle=(sampler is None),
                                sampler=sampler)

        for index, (X_batch, y_batch, idx) in enumerate(tqdm(dataloader)):
            # for index in tqdm(range(0, len(X_test), batch_size)):
            # X_batch = X_test[index:index + batch_size]
            # y_batch = y_test[index:index + batch_size]
            # X_batch, y_batch = T2VDataset(
            #     X_batch, y_batch, vocab, device, config).return_all()
            X_batch_unpad = Xp_unpad[idx]
            Xn, yn = NegSample(
                X_batch_unpad, config).generate_neg(vocab, device)
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
                logger.info(f"Epoch: {epoch+1}/{epochs}")
                logger.info(f'TEST: {index}')
                logger.info(
                    f"INPUT TABLE:\n{print_table(total_inputs[0].cpu().data.numpy(),vocab)}")
                logger.info(
                    f'GOLD LABEL: {y_batch_total[0].cpu().data.numpy()}')
                logger.info(f'PREDICTED LABEL: {y_p[0].cpu().data.numpy()}\n')
                # step = (epoch * len(X_train) + index) / batch_size
                step = epoch * (len(test_dataset_p) / batch_size) + index
                accuracy_batch = accuracy_score(y_batch_total.cpu().data.numpy(),
                                                y_p.cpu().data.numpy())
                train_writer.add_scalar('Batch_lvl/Loss', loss.item(), step)
                train_writer.add_scalar(
                    'Batch_lvl/Accuracy', accuracy_batch, step)

        accuracy = accuracy_score(y_actual_test, y_pred_test)
        # precision = precision_score(y_actual_test, y_pred_test)
        # recall = recall_score(y_actual_test, y_pred_test)
        # f1 = f1_score(y_actual_test, y_pred_test)

        test_writer.add_scalar('Loss', loss_per_epoch, epoch)
        test_writer.add_scalar('Accuracy', accuracy, epoch)
        # test_writer.add_scalar('F1', f1, epoch)
        # test_writer.add_scalar('Precision', precision, epoch)
        # test_writer.add_scalar('Recall', recall, epoch)
        # logger.info(
        #     f"Testing - Loss : {loss_per_epoch}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}")
        logger.info(
            f"Testing - Epoch: {epoch+1}, Loss: {loss_per_epoch}, Accuracy: {accuracy}")

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

    torch.save(model.state_dict(), os.path.join(
        output_dir, 'model_state_dict.pt'))
    torch.save(model, os.path.join(
        output_dir, 'model.pt'))
    end_time_total = time.time() - start_time_total
    logger.info(f"Time spent total : {str(end_time_total/3600)} hrs\n")
    # plot_ndcg_epochs(ndcg_scores_total, output_dir, config)

    train_writer.close()
    test_writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="path to configuration file (toml format)")
    parser.add_argument("-g", "--gpu",
                        help="gpu number to run with")
    parser.add_argument("-m", "--model_type",
                        help="model type to be trained")
    parser.add_argument("--comment",
                        help="additional comments for simulation to be run.")
    parser.add_argument("-sd", "--smaller_data", default=None, type=int,
                        help="Size for smaller data to be tested")
    parser.add_argument("--use_checkpoint", default=None, type=str,
                        help="Use checkpoint or not")
    parser.add_argument("--distributed", action='store_true',
                        help="Distributed training or not")
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    output_dir, config = setup_simulation(args)

    # seed setting
    seed = config['model_params']['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logger.info(f"python {' '.join(sys.argv)}")
    fit(config=(args, config), output_dir=output_dir)
