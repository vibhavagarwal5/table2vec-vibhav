import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from itertools import chain, zip_longest
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger("app")


def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def setup_loggers(output_dir):
    # create handlers
    app_fh = logging.FileHandler(os.path.join(output_dir, "app.log"))
    app_sh = logging.StreamHandler(sys.stdout)

    # create formatters and add to handlers
    file_formatter = logging.Formatter("%(levelname)s:%(message)s")
    stream_formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    app_fh.setFormatter(file_formatter)
    app_sh.setFormatter(stream_formatter)

    # add handlers to loggers
    app_logger = logging.getLogger("app")
    app_logger.addHandler(app_fh)
    app_logger.addHandler(app_sh)

    # set log levels
    app_logger.setLevel('INFO')
    app_logger.propogate = False


def load_config_args(args):
    config = Config()
    if args.config:
        config.load(args.config)

    config.add_to_config("comment", args.comment)
    config.add_to_config("gpu", args.gpu)
    config.add_to_config("distributed", args.distributed)
    config['model_props']['type'] = args.model_type
    config.add_to_config("model_props", config['model_props'])
    return config


def setup_simulation(args):
    config = load_config_args(args)

    # create output directory
    t = datetime.today()
    output_dir = os.path.join(
        config["output"], f"{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"
    )
    make_dirs(output_dir)

    setup_loggers(output_dir)
    logger.info(f"Created output directory {output_dir}")

    # save options
    config_path = os.path.join(output_dir, "config.toml")
    config.save(config_path)
    logger.info(f"Saved configuration options at {config_path}")

    return output_dir, config


def loadpkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def savepkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def flatten_1_deg(arr):
    return list(chain(*arr))


def pool_fn(fn, inp, processes):
    p = Pool(processes=processes)
    out = p.map(fn, inp)
    p.close()
    p.join()
    return out


def mp(df, func, num_partitions):
    df_split = np.array_split(df, num_partitions)
    p = Pool(num_partitions)
    df = pd.concat(p.map(func, df_split))
    # with Pool(num_partitions) as p:
    #     df = pd.concat(tqdm(p.imap(func, df_split), total=len(df_split)))
    p.close()
    p.join()
    return df


def print_tableIDs(table, vocab, logger_print='print'):
    print_fn = logger.info if logger_print == 'logger' else print
    i2w = {i: w for i, w in enumerate(vocab)}
    table = table.tolist()
    for r in table:
        for c in r:
            for i, w in enumerate(c):
                c[i] = i2w[int(w)]
        print_fn(r)


def print_table(table, logger_print='print'):
    print_fn = logger.info if logger_print == 'logger' else print
    for r in table:
        print_fn(r)
