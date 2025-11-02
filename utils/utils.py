import torch
import random
import os
import re
import numpy as np
import yaml

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def set_gpu_devices(gpu_id):
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def _convert_types(obj):

    if isinstance(obj, dict):
        return {k: _convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_types(x) for x in obj]
    elif isinstance(obj, str):
        try:
            if any(c in obj for c in [".", "e", "E"]):
                return float(obj)
            elif obj.isdigit():
                return int(obj)
        except ValueError:
            pass
        return obj
    else:
        return obj

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return _convert_types(dict)


def multisplit(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)