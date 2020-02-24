import argparse
import logging
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import models
import utils
from dataset import T2VDataset
from training import fit, make_writer
from trec import TREC_data_prep, TREC_model
from TREC_score import ndcg_pipeline
from utils import loadpkl

logger = logging.getLogger("app")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to configuration file (toml format)")
    parser.add_argument(
        "-c", "--cuda_no", help="gpu number to run with")
    parser.add_argument(
        "-m", "--model_type", help="model type to be trained")
    parser.add_argument(
        "--comment", help="additional comments for simulation to be run."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    output_dir, config = utils.setup_simulation(args)
    model_params = config['model_params']
    input_files = config['input_files']
    trec_config = config['trec']

    torch.manual_seed(model_params['seed'])

    Xp = loadpkl(input_files['Xp_path'])
    yp = loadpkl(input_files['yp_path'])
    logger.info(f"Xp.shape: {Xp.shape}, yp.shape: {yp.shape}")
    # Xn = loadpkl(input_files['Xn_path'])
    # yn = loadpkl(input_files['yn_path'])
    # logger.info(f"Xn.shape: {Xn.shape}, yn.shape: {yn.shape}")
    vocab = loadpkl(input_files['vocab_path'])
    logger.info(f"len(vocab): {len(vocab)}")

    train_writer = make_writer(output_dir, 'train', config)
    test_writer = make_writer(output_dir, 'test', config)

    device = torch.device(
        f"cuda:{args.cuda_no}" if torch.cuda.is_available() else 'cpu')

    model = models.create_model(config['model_props']['type'], params=(
        len(vocab), model_params['embedding_dim'], device, config))
    # model = Table2Vec(len(vocab), model_params['embedding_dim'], device)
    model = model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    fit(
        model=model,
        pos_sample=(Xp, yp),
        neg_sample=(),
        # neg_sample=(Xn, yn),
        vocab=vocab,
        loss_fn=loss_function,
        opt=optimizer,
        config=config,
        output_dir=output_dir,
        writers=(train_writer, test_writer),
        device=device)

    train_writer.close()
    test_writer.close()
