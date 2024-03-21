# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine import Registry
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import DATASETS as MMENGINE_DATASETS

MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['estimator.models'])
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['estimator.datasets'])
