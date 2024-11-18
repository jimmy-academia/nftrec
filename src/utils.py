import re
import json
import torch
import argparse

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        # json.dump(dictionary, f, indent=4)
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def set_seed(seed):
    import os
    import random
    # import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_verbose(verbose):
    # usages: logging.warning; logging.error, logging.info, logging.debug
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],  # Print to terminal
    )
    # Test the logging configuration
    # logging.warning("Logging setup complete - WARNING test")
    # logging.info("Logging setup complete - INFO test")
    # logging.debug("Logging setup complete - DEBUG test")

def deep_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: deep_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_cpu(v) for v in obj]
    else:
        return obj

def deep_to_pylist(obj):
    if isinstance(obj, torch.Tensor):
        # If it's a scalar tensor, use item()
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {k: deep_to_pylist(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_pylist(v) for v in obj]
    else:
        return obj
    
def deep_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: deep_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_device(v, device) for v in obj]
    else:
        return obj
    
def torch_cleansave(obj, path):
    obj = deep_to_cpu(obj)
    torch.save(obj, path)

def torch_cleanload(path, device):
    obj = torch.load(path, weights_only=True)
    return deep_to_device(obj, device)
