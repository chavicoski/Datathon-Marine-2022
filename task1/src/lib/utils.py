"""Generic utilities"""
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 1234):
    """Sets the random seed for all the librarie used"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
