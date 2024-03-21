import time
import torch

from mmengine.dist import (broadcast, get_dist_info, init_dist, is_distributed, get_local_rank)
from mmengine.utils.dl_utils import (set_multi_processing)
def setup_env(env_cfg, distributed, launcher):
    """Setup environment.

    An example of ``env_cfg``::

        env_cfg = dict(
            cudnn_benchmark=True,
            mp_cfg=dict(
                mp_start_method='fork',
                opencv_num_threads=0
            ),
            dist_cfg=dict(backend='nccl', timeout=1800),
            resource_limit=4096
        )

    Args:
        env_cfg (dict): Config for setting environment.
    """
    if env_cfg.get('cudnn_benchmark'):
        torch.backends.cudnn.benchmark = True

    mp_cfg: dict = env_cfg.get('mp_cfg', {})
    set_multi_processing(**mp_cfg, distributed=distributed)

    # init distributed env first, since logger depends on the dist info.
    if distributed and not is_distributed():
        dist_cfg: dict = env_cfg.get('dist_cfg', {})
        init_dist(launcher, **dist_cfg)

    _rank, _world_size = get_dist_info()
    # _local_rank = get_local_rank()

    timestamp = torch.tensor(time.time(), dtype=torch.float64)
    # broadcast timestamp from 0 process to other processes
    broadcast(timestamp)
    _timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
    return _rank, _world_size, _timestamp

