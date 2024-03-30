import os
import os.path as osp
import argparse
import torch
import time
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger

from estimator.utils import RunnerInfo, setup_env, log_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.tester import Tester
from estimator.models.patchfusion import PatchFusion
from mmengine import print_log

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', 
        help='the dir to save logs and models', 
        default=None)
    parser.add_argument(
        '--test-type',
        type=str,
        default='normal',
        help='evaluation type')
    parser.add_argument(
        '--ckp-path',
        type=str,
        help='ckp_path')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='save colored prediction & depth predictions')
    parser.add_argument(
        '--cai-mode', 
        type=str,
        default='m1',
        help='m1, m2, or rx')
    parser.add_argument(
        '--process-num',
        type=int, default=2,
        help='batchsize number for inference')
    parser.add_argument(
        '--tag',
        type=str, default='',
        help='infer_infos')
    parser.add_argument(
        '--gray-scale',
        action='store_true',
        default=False,
        help='use gray-scale color map')
    parser.add_argument(
        '--image-raw-shape',
        nargs='+', default=[2160, 3840])
    parser.add_argument(
        '--patch-split-num',
        nargs='+', default=[4, 4])
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

    image_raw_shape=[int(num) for num in args.image_raw_shape]
    patch_split_num=[int(num) for num in args.patch_split_num]
        
    # load config
    cfg = Config.fromfile(args.config)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use ckp path as default work_dir if cfg.work_dir is None
        if '.pth' in args.ckp_path:
            args.work_dir = osp.dirname(args.ckp_path)
        else:
            args.work_dir = osp.join('work_dir', args.ckp_path.split('/')[1])
        cfg.work_dir = args.work_dir
        
    mkdir_or_exist(cfg.work_dir)
    cfg.ckp_path = args.ckp_path
    
    # fix seed
    seed = cfg.get('seed', 5621)
    fix_random_seed(seed)
    
    # start dist training
    if cfg.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
        env_cfg = cfg.get('env_cfg')
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # build dataloader
    if args.test_type == 'consistency':
        dataloader_config = cfg.val_consistency_dataloader
        dataset = build_dataset(cfg.val_consistency_dataloader.dataset)
    elif args.test_type == 'normal':
        dataloader_config = cfg.val_dataloader
        dataset = build_dataset(cfg.val_dataloader.dataset)
    elif args.test_type == 'test_in':
        dataloader_config = cfg.test_in_dataloader
        dataset = build_dataset(cfg.test_in_dataloader.dataset)
    elif args.test_type == 'test_out':
        dataloader_config = cfg.test_out_dataloader
        dataset = build_dataset(cfg.test_out_dataloader.dataset)
    elif args.test_type == 'general':
        dataloader_config = cfg.general_dataloader
        dataset = build_dataset(cfg.general_dataloader.dataset)
    else:
        dataloader_config = cfg.val_dataloader
        dataset = build_dataset(cfg.val_dataloader.dataset)
    
    dataset.image_resolution = image_raw_shape
    
    # extract experiment name from cmd
    config_path = args.config
    exp_cfg_filename = config_path.split('/')[-1].split('.')[0]
    ckp_name = args.ckp_path.replace('/', '_').replace('.pth', '')
    dataset_name = dataset.dataset_name
    # log_filename = 'eval_{}_{}_{}_{}.log'.format(timestamp, exp_cfg_filename, ckp_name, dataset_name)
    log_filename = 'eval_{}_{}_{}_{}_{}.log'.format(exp_cfg_filename, args.tag, ckp_name, dataset_name, timestamp)
    
    # prepare basic text logger
    log_file = osp.join(args.work_dir, log_filename)
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
    runner_info.save = args.save
    runner_info.log_filename = log_filename
    runner_info.gray_scale = args.gray_scale
    
    if runner_info.save:
        mkdir_or_exist(args.work_dir)
        runner_info.work_dir = args.work_dir
    # log_env(cfg, env_cfg, runner_info, logger)
    
    # build model
    if '.pth' in cfg.ckp_path:
        model = build_model(cfg.model)
        print_log('Checkpoint Path: {}. Loading from a local file'.format(cfg.ckp_path), logger='current')
        if hasattr(model, 'load_dict'):
            print_log(model.load_dict(torch.load(cfg.ckp_path)['model_state_dict']), logger='current')
        else:
            print_log(model.load_state_dict(torch.load(cfg.ckp_path)['model_state_dict'], strict=True), logger='current')
    else:
        print_log('Checkpoint Path: {}. Loading from the huggingface repo'.format(cfg.ckp_path), logger='current')
        assert cfg.ckp_path in \
            ['Zhyever/patchfusion_depth_anything_vits14', 
             'Zhyever/patchfusion_depth_anything_vitb14', 
             'Zhyever/patchfusion_depth_anything_vitl14', 
             'Zhyever/patchfusion_zoedepth'], 'Invalid model name'
        model = PatchFusion.from_pretrained(cfg.ckp_path)
    model.eval()
    
    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runner_info.rank], output_device=runner_info.rank,
                                                          find_unused_parameters=cfg.get('find_unused_parameters', False))
        logger.info(model)
    else:
        model.cuda()
        
    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)

    # build tester
    tester = Tester(
        config=cfg,
        runner_info=runner_info,
        dataloader=val_dataloader,
        model=model)
    
    if args.test_type == 'consistency':
        tester.run_consistency()
    else:
        tester.run(args.cai_mode, process_num=args.process_num, image_raw_shape=image_raw_shape, patch_split_num=patch_split_num)

if __name__ == '__main__':
    main()