import argparse
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
from dataset import T2VDataset, collate_fn
from trec import TREC_data_prep, TREC_model
from TREC_score import ndcg_pipeline
from utils import (flatten_1_deg, loadpkl, make_dirs, mp, print_tableIDs,
                   savepkl, setup_simulation)

logger = logging.getLogger("app")
prev_embd = None


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info('Validation loss decreased ({0} --> {1}).  Saving model ...'.format(
                self.val_loss_min, val_loss))
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


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


def train(config, output_dir):
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
    batch_size = int(config['model_params']['batch_size'] / 2)
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
        params=(len(vocab), config['model_params']['embedding_dim']))
    # Use checkpoint load if specified
    if args.use_checkpoint is not None:
        model.load_state_dict(torch.load(args.use_checkpoint))
        logger.info('Using checkpoint from : {}'.format(args.use_checkpoint))
    else:
        logger.info('Using fresh model')
    model = model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=config['model_params']['opt_lr'])
    scheduler = optim.lr_scheduler.MultiplicativeLR(opt,
                                                    lr_lambda=lambda epoch: 0.66)
    # DistributedDataParallel
    if args.distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)

    # Creating +ve dataset
    dataset_p = T2VDataset(Xp, yp, vocab, config)

    # Splitting the +ve dataset into train and test sets
    train_size = int(0.7 * len(dataset_p))
    test_size = len(dataset_p) - train_size
    train_dataset_p, test_dataset_p = random_split(
        dataset_p, [train_size, test_size])
    # X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    #     Xp, yp, range(len(Xp)), train_size=0.7, random_state=config['model_params']['seed'])
    # X_train_unpad, X_test_unpad = Xp_unpad[idx_train], Xp_unpad[idx_test]

    # Creating training dataloader
    sampler_train = None
    if args.distributed:
        sampler_train = DistributedSampler(train_dataset_p)
    dataloader_train = DataLoader(train_dataset_p,
                                  batch_size=batch_size,
                                  shuffle=(sampler_train is None),
                                  sampler=sampler_train,
                                  collate_fn=lambda batch: collate_fn(batch, Xp_unpad,
                                                                      config, vocab))

    # Creating testing dataloader
    sampler_test = None
    if args.distributed:
        sampler_test = DistributedSampler(test_dataset_p)
    dataloader_test = DataLoader(test_dataset_p,
                                 batch_size=batch_size,
                                 shuffle=(sampler_test is None),
                                 sampler=sampler_test,
                                 collate_fn=lambda batch: collate_fn(batch, Xp_unpad,
                                                                     config, vocab))

    early_stopping = EarlyStopping(patience=3,
                                   verbose=True)
    start_time_total = time.time()
    for epoch in range(1, epochs + 1):
        start_time_epoch = time.time()

        ''' Training '''
        loss_per_epoch = []
        correct = 0
        for index, (X_batch, y_batch, _) in enumerate(tqdm(dataloader_train)):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            y_p = torch.round(y_pred)
            correct += (y_p == y_batch).float().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_per_epoch.append(loss.item())

            if index % config['model_props']['print_every'] == 0:
                logger.info(f"Epoch: {epoch}/{epochs}")
                logger.info(f'TRAIN: {index}')
                logger.info("INPUT TABLE:\n{0}".format(
                    print_tableIDs(X_batch[0], vocab, 'logger')))
                logger.info('GOLD LABEL: {0}'.format(y_batch[0].item()))
                logger.info('PREDICTED LABEL: {0}'.format(y_p[0].item()))
                step = epoch * len(dataloader_train) + index
                accuracy_batch = 100 * correct.item() / ((index + 1) * len(y_batch))
                logger.info('Accuracy: {0}\n'.format(accuracy_batch))
                train_writer.add_scalar('Batch_lvl/Loss',
                                        np.average(loss_per_epoch), step)
                train_writer.add_scalar('Batch_lvl/Accuracy',
                                        accuracy_batch, step)

        accuracy = 100 * correct.item() / (len(y_batch) * len(dataloader_train))
        train_writer.add_scalar('Loss', np.average(loss_per_epoch), epoch)
        train_writer.add_scalar('Accuracy', accuracy, epoch)
        logger.info("Training - Epoch: {0}, Loss: {1}, Accuracy: {2}".format(
            epoch, np.average(loss_per_epoch), accuracy))

        '''--------------------------------------------------------------------------------------'''

        ''' Testing '''
        # loss_per_epoch, y_actual_test, y_pred_test = [], [], []
        loss_per_epoch = []
        correct = 0
        for index, (X_batch, y_batch, _) in enumerate(tqdm(dataloader_test)):
            #     # for index in tqdm(range(0, len(X_test), batch_size)):
            #     # X_batch = X_test[index:index + batch_size]
            #     # y_batch = y_test[index:index + batch_size]
            #     # X_batch, y_batch = T2VDataset(
            #     #     X_batch, y_batch, vocab, device, config).return_all()
            #     X_batch_unpad = Xp_unpad[idx]
            #     Xn, yn = NegSample(
            #         X_batch_unpad, config).generate_neg(vocab, device)
            #     total_inputs = torch.cat((X_batch, Xn), dim=0)
            #     y_batch_total = torch.cat((y_batch, yn), dim=0)
            #     shuffle = torch.randperm(len(total_inputs))
            #     total_inputs = total_inputs[shuffle]
            #     y_batch_total = y_batch_total[shuffle]

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # y_actual_test += list(y_batch.cpu().data.numpy())
            y_p = torch.round(y_pred)
            correct += (y_p == y_batch).float().sum()
            # y_pred_test += list(y_p.cpu().data.numpy())
            loss_per_epoch.append(loss.item())

            if index % config['model_props']['print_every'] == 0:
                logger.info(f"Epoch: {epoch}/{epochs}")
                logger.info(f'TEST: {index}')
                logger.info("INPUT TABLE:\n{0}".format(
                    print_tableIDs(X_batch[0], vocab, 'logger')))
                logger.info('GOLD LABEL: {0}'.format(y_batch[0].item()))
                logger.info('PREDICTED LABEL: {0}'.format(y_p[0].item()))
                # logger.info("INPUT TABLE:\n{0}".format(
                #     print_tableIDs(X_batch[0].cpu().data.numpy(),
                #                    vocab, 'logger')))
                # logger.info('GOLD LABEL: {0}'.format(
                #     y_batch[0].cpu().data.numpy()))
                # logger.info('PREDICTED LABEL: {0}\n'.format(
                #     y_p[0].cpu().data.numpy()))
                # step = (epoch * len(X_train) + index) / batch_size
                step = epoch * len(dataloader_test) + index
                # accuracy_batch = accuracy_score(y_batch.cpu().data.numpy(),
                #                                 y_p.cpu().data.numpy())
                accuracy_batch = 100 * correct.item() / ((index + 1) * len(y_batch))
                logger.info('Accuracy: {0}\n'.format(accuracy_batch))
                test_writer.add_scalar('Batch_lvl/Loss',
                                       np.average(loss_per_epoch), step)
                test_writer.add_scalar('Batch_lvl/Accuracy',
                                       accuracy_batch, step)

        accuracy = 100 * correct.item() / (len(y_batch) * len(dataloader_train))
        # accuracy = accuracy_score(y_actual_test, y_pred_test)
        # precision = precision_score(y_actual_test, y_pred_test)
        # recall = recall_score(y_actual_test, y_pred_test)
        # f1 = f1_score(y_actual_test, y_pred_test)

        test_writer.add_scalar('Loss', np.average(loss_per_epoch), epoch)
        test_writer.add_scalar('Accuracy', accuracy, epoch)
        # test_writer.add_scalar('F1', f1, epoch)
        # test_writer.add_scalar('Precision', precision, epoch)
        # test_writer.add_scalar('Recall', recall, epoch)
        # logger.info(
        #     f"Testing - Loss : {loss_per_epoch/len(dataloader_test)}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}")
        logger.info("Testing - Epoch: {0}, Loss: {1}, Accuracy: {2}".format(
            epoch, np.average(loss_per_epoch), accuracy))

        ''' After train and test loop...'''
        early_stopping(np.average(loss_per_epoch), model,
                       os.path.join(output_dir, f"model_{epoch}.pt"))
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch:{epoch}")
            break
        if early_stopping.counter == 0:
            for param_group in opt.param_groups:
                logger.info("Intial LR:{0}".format(param_group['lr']))
            scheduler.step()
            for param_group in opt.param_groups:
                logger.info("After scheduler update, LR:{0}".format(
                    param_group['lr']))

        embeddings = model.embeddings
        if prev_embd is None:
            prev_embd = embeddings.weight
        else:
            curr_embd = embeddings.weight
            logger.info("Are the embeddings same as the previous one? -> {0}".format(
                torch.equal(prev_embd, curr_embd)))
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

    # torch.save(model, os.path.join(output_dir, 'model.pt'))
    end_time_total = time.time() - start_time_total
    logger.info("Time spent total : {0} hrs\n".format(
        str(end_time_total / 3600)))
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
    torch.backends.cudnn.deterministic = True

    logger.info(f"python {' '.join(sys.argv)}")
    train(config=(args, config), output_dir=output_dir)
