# Copyright 2023 Toyota Research Institute.  All rights reserved.

from argparse import Namespace

import numpy as np
import torch


def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)


def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor


def is_tuple(data):
    """Checks if data is a tuple."""
    return isinstance(data, tuple)


def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list) or isinstance(data, torch.nn.ModuleList)


def is_double_list(data):
    """Checks if data is a double list (list of lists)"""
    return is_list(data) and len(data) > 0 and is_list(data[0])


def is_dict(data):
    """Checks if data is a dictionary."""
    return isinstance(data, dict) or isinstance(data, torch.nn.ModuleDict)


def is_module_dict(data):
    """Checks if data is a dictionary."""
    return isinstance(data, torch.nn.ModuleDict)


def is_str(data):
    """Checks if data is a string."""
    return isinstance(data, str)


def is_int(data):
    """Checks if data is an integer."""
    return isinstance(data, int)


def is_seq(data):
    """Checks if data is a list or tuple."""
    return is_tuple(data) or is_list(data)


def is_double_seq(data):
    """Checks if data is a double list (list of lists)"""
    return is_seq(data) and len(data) > 0 and is_seq(data[0])


def is_namespace(data):
    """Check if data is a Namespace"""
    return isinstance(data, Namespace)


def exists(data):
    """Check if data exists (it is not None)"""
    return data is not None