import argparse
import logging
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

import models
from training import fit, make_writer
from utils import loadpkl, setup_simulation

logger = logging.getLogger("app")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to configuration file (toml format)")
    parser.add_argument(
        "-g", "--gpu", help="gpu number to run with")
    parser.add_argument(
        "-m", "--model_type", help="model type to be trained")
    parser.add_argument(
        "--comment", help="additional comments for simulation to be run."
    )
    parser.add_argument(
        "-sd", "--smaller_data", default=None, type=int, help="Size for smaller data to be tested"
    )
    parser.add_argument(
         "--use_checkpoint", default=None, type=str, help="Use checkpoint or not"
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    output_dir, config = setup_simulation(args)
    model_params = config['model_params']
    input_files = config['input_files']
    trec_config = config['trec']

    torch.manual_seed(model_params['seed'])
    np.random.seed(model_params['seed'])

    Xp = loadpkl(input_files['Xp_path'])
    yp = np.ones((len(Xp), 1))
    if args.smaller_data is not None:
        Xp = Xp[:args.smaller_data]
        yp = yp[:args.smaller_data]
    # yp = loadpkl(input_files['yp_path'])
    logger.info(f"Xp.shape: {Xp.shape}, yp.shape: {yp.shape}")
    # Xn = loadpkl(input_files['Xn_path'])
    # yn = loadpkl(input_files['yn_path'])
    # logger.info(f"Xn.shape: {Xn.shape}, yn.shape: {yn.shape}")
    vocab = loadpkl(input_files['vocab_path'])
    logger.info(f"len(vocab): {len(vocab)}")

    fit(
        pos_sample=(Xp, yp),
        neg_sample=(),
        # neg_sample=(Xn, yn),
        vocab=vocab,
        config=(args, config),
        output_dir=output_dir,
    )
