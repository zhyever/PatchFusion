
import random
import numpy
import torch
from typing import List, Optional, Sequence, Tuple, Union
from mmengine.config import ConfigDict
from mmengine.utils.dl_utils import collect_env
from collections import OrderedDict


ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

def fix_random_seed(seed: int):

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = False # speed up maybe
    torch.backends.cudnn.benchmark = True

def log_env(cfg, env_cfg, runner_info, logger):
    """Logging environment information of the current task.

    Args:
        env_cfg (dict): The environment config of the runner.
    """
    # Collect and log environment information.
    env = collect_env()
    runtime_env = OrderedDict()
    runtime_env.update(env_cfg)
    runtime_env['seed'] = runner_info.seed
    runtime_env['Distributed launcher'] = runner_info.launcher
    runtime_env['Distributed training'] = runner_info.distributed
    runtime_env['GPU number'] = runner_info.world_size

    env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
                                        for k, v in env.items())
    runtime_env_info = '\n    ' + '\n    '.join(
        f'{k}: {v}' for k, v in runtime_env.items())
    dash_line = '-' * 60
    logger.info('\n' + dash_line + '\nSystem environment:' +
                    env_info + '\n'
                    '\nRuntime environment:' + runtime_env_info + '\n' +
                    dash_line + '\n')

    if cfg._cfg_dict:
        logger.info(f'Config:\n{cfg.pretty_text}')