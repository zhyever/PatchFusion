import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger

from estimator.utils import RunnerInfo, setup_env, log_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode')
    parser.add_argument(
        '--log-name',
        type=str, default='',
        help='log_name for wandb')
    parser.add_argument(
        '--tags',
        type=str, default='',
        help='tags for wandb')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--seed',
        type=int, default=621,
        help='for debug')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    
    args = parse_args()
    
    # if args.debug:
    #     torch.autograd.set_detect_anomaly(True) # for debug

    # load config
    cfg = Config.fromfile(args.config)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    cfg.work_dir = args.work_dir
    cfg.work_dir = osp.join(cfg.work_dir, args.log_name)
    
    mkdir_or_exist(cfg.work_dir)
    cfg.debug = args.debug
    cfg.log_name = args.log_name
    tags = args.tags
    if ',' in tags:
        tag_list = tags.split(',')
    else:
        tag_list = [tags]
    cfg.tags = tag_list
    
    # fix seed
    seed = args.seed
    fix_random_seed(seed)
    
    # start dist training
    if cfg.launcher == 'none':
        distributed = False
    else:
        distributed = True
    env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
    rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # prepare basic text logger
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', timestamp)
    log_cfg.setdefault('logger_name', 'patchstitcher')
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)
    
    # save some information useful during the training
    runner_info = RunnerInfo()
    runner_info.config = cfg # ideally, cfg should not be changed during process. information should be temp saved in runner_info
    runner_info.logger = logger # easier way: use print_log("infos", logger='current')
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    
    # start wandb
    if runner_info.rank == 0 and cfg.debug == False:
        wandb.init(
            project=cfg.project, 
            name=cfg.log_name+"_"+runner_info.timestamp, 
            tags=cfg.tags, 
            dir=runner_info.work_dir,
            config=cfg, # have a test
            settings=wandb.Settings(start_method="fork"))
        
        wandb.define_metric("Val/step")
        wandb.define_metric("Val/*", step_metric="Val/step")
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/*", step_metric="Train/step")
    
    log_env(cfg, env_cfg, runner_info, logger)
    
    # resume training (future)
    cfg.resume = args.resume
    
    # build model
    model = build_model(cfg.model)
    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        if cfg.get('convert_syncbn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runner_info.rank], output_device=runner_info.rank,
                                                          find_unused_parameters=cfg.get('find_unused_parameters', False))
        logger.info(model)
    else:
        model = model.cuda(runner_info.rank)
        logger.info(model)
        
    # build dataloader
    dataset = build_dataset(cfg.train_dataloader.dataset)
    if runner_info.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.train_dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=train_sampler)


    dataset = build_dataset(cfg.val_dataloader.dataset)
    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)

    # everything is ready, start training. But before that, save your config!
    cfg.dump(osp.join(cfg.work_dir, 'config.py'))
    
    # build trainer
    trainer = Trainer(
        config=cfg,
        runner_info=runner_info,
        train_sampler=train_sampler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model)
    
    trainer.run()
    wandb.finish()

if __name__ == '__main__':
    main()