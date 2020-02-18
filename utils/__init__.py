import pickle
import argparse
import sys
import os
import logging
from datetime import datetime

from .config import Config

logger = logging.getLogger("app")


def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


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
    config.add_to_config("cuda_no", args.cuda_no)
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
