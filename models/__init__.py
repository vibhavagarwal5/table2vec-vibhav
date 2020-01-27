import logging
import importlib
import torch.nn as nn
import inspect

logger = logging.getLogger("app")


def create_model(model_type, params):
    if model_type is None:
        raise Exception(f"Model type is None.")

    module = importlib.import_module(f"models.{model_type}")
    model_class = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            model_class = obj
            logger.info(model_class)
            break

    if model_class is None:
        raise Exception(f"Model type {model_type} is not valid.")

    model = model_class(*params)
    return model
