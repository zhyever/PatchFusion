# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from estimator.registry import DATASETS

def build_dataset(cfg):
    """Build backbone."""
    return DATASETS.build(cfg)