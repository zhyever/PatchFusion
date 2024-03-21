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

from transformers import PretrainedConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--ckp-path',
        type=str,
        help='ckp_path')
    parser.add_argument(
        '--save-path',
        type=str,
        help='ckp_path')
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

    # load config
    cfg = Config.fromfile(args.config)
    cfg.ckp_path = args.ckp_path

    # folder_name = os.path.dirname(args.save_path)
    # print(folder_name)
    # exit(100)
    
    # build model
    
    model = build_model(cfg.model)
    print_log('Checkpoint Path: {}'.format(cfg.ckp_path), logger='current')
    if hasattr(model, 'load_dict'):
        print_log(model.load_dict(torch.load(cfg.ckp_path)['model_state_dict']), logger='current')
    else:
        print_log(model.load_state_dict(torch.load(cfg.ckp_path)['model_state_dict'], strict=True), logger='current')
    model.eval()
    
    model.save_pretrained(args.save_path)
    model.config.to_json_file(os.path.join(args.save_path, "config.json"))
    
    
    # model = PatchFusion.from_pretrained('Zhyever/patchfusion_depth_anything_vits14')


if __name__ == '__main__':
    main()