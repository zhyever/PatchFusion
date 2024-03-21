import os
import cv2
import numpy as np
import torch
import imageio
import sys
import re

from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

from estimator.registry import DATASETS
from estimator.utils import get_boundaries, compute_metrics
from estimator.datasets.transformers import aug_color, aug_flip, to_tensor, random_crop, aug_rotate
from estimator.datasets.u4k_dataset import UnrealStereo4kDataset
from estimator.datasets.utils import readPFM

from zoedepth.models.base_models.midas import Resize
from depth_anything.transform import Resize as ResizeDA

def read_image(path, dataset_name, image_resolution=(2160, 3840)):
    if dataset_name == 'u4k':
        img = np.fromfile(open(path, 'rb'), dtype=np.uint8).reshape(2160, 3840, 3) / 255.0
        img = img.astype(np.float32)[:, :, ::-1].copy()
    elif dataset_name == 'mid':
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), image_resolution, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0).numpy()
    elif dataset_name == 'cityscapes':
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = Image.open(path).convert("RGB")
        img = np.asarray(img).astype(np.float32).copy()
        img = img / 255.0
    else:
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), image_resolution, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0).numpy()
    
    return img
                            
class Images:
    def __init__(self, root_dir, files, index, dataset_name=None):
        self.root_dir = root_dir
        name = files[index]
        self.dataset_name = dataset_name
        self.rgb_image = read_image(os.path.join(self.root_dir, name), dataset_name)
        name = name.replace(".jpg", "")
        name = name.replace(".png", "")
        name = name.replace(".jpeg", "")
        self.name = name
        
class DepthMap:
    def __init__(self, root_dir, files, index, dataset_name, pred=False):
        self.root_dir = root_dir
        name = files[index]

        gt_path = os.path.join(self.root_dir, name)
        if dataset_name == 'u4k':
            depth_factor = gt_path.replace('val_gt', 'val_factor')
            depth_factor = depth_factor.replace('.npy', '.txt')
            with open(depth_factor, 'r') as f:
                df = f.readline()
            df = float(df)

            gt_disp = np.load(gt_path, mmap_mode='c')
            gt_disp = gt_disp.astype(np.float32)
            edges = get_boundaries(gt_disp, th=1, dilation=0)

            gt_depth = df/gt_disp
            self.gt = gt_depth
            self.edge = edges

        elif dataset_name == 'gta':
            gt_depth = imageio.imread(gt_path)
            gt_depth = np.array(gt_depth).astype(np.float32) / 256
            edges = get_boundaries(gt_depth, th=1, dilation=0)
            self.gt = gt_depth
            self.edge = edges
        
        elif dataset_name == 'eth3d':
            height, width = 4032, 6048
            # depth = cv2.imread(gt_path)
            depth = np.fromfile(gt_path, dtype=np.float32).reshape(height, width)
            depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
            depth = depth.astype(np.float32)
            edges = get_boundaries(depth, th=1, dilation=0)
            self.gt = depth
            self.edge = edges
    
        elif dataset_name == 'mid':
            
            depth_factor = gt_path.replace('gts', 'calibs')
            depth_factor = depth_factor.replace('.pfm', '.txt')
            with open(depth_factor, 'r') as f:
                ext_l = f.readlines()
            cam_info = ext_l[0].strip()
            cam_info_f = float(cam_info.split(' ')[0].split('[')[1])
            base = float(ext_l[3].strip().split('=')[1])
            doffs = float(ext_l[2].strip().split('=')[1])
            depth_factor = base * cam_info_f
            
            disp_gt, scale = readPFM(gt_path)
            disp_gt = disp_gt.astype(np.float32)

            disp_gt_copy = disp_gt.copy()
            disp_gt = disp_gt
            invalid_mask = disp_gt == np.inf
            
            depth_gt = depth_factor / (disp_gt + doffs)
            depth_gt = depth_gt / 1000
            depth_gt[invalid_mask] = 0 # set to a invalid number
            disp_gt_copy[invalid_mask] = 0
            edges = get_boundaries(disp_gt_copy, th=1, dilation=0)

            self.gt = depth_gt
            self.edge = edges
        
        elif dataset_name == 'cityscapes':
            img_d = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256
            depth_gt = (0.209313 * 2262.52) / img_d
            depth_gt = np.nan_to_num(depth_gt, posinf=0., neginf=0., nan=0.)
            
            depth_gt = depth_gt.astype(np.float32)
            edges = get_boundaries(depth_gt, th=1, dilation=0)

            self.gt = depth_gt
            self.edge = edges
        
        else:
            raise NotImplementedError
        
        name = name.replace(".npy", "") # u4k
        name = name.replace(".exr", "") # gta
        self.name = name

@DATASETS.register_module()
class ImageDataset(UnrealStereo4kDataset):
    def __init__(
        self,
        rgb_image_dir,
        mode='',
        min_depth=1e-3,
        max_depth=80,
        gt_dir=None,
        # process_shape=[384, 512],
        dataset_name='',
        network_process_size=(384, 512),
        resize_mode='zoe'):
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mode = mode # Default run in a CAI mode
        
        self.rgb_image_dir = rgb_image_dir
        self.files = sorted(os.listdir(self.rgb_image_dir))
        self.gt_dir = gt_dir
        self.dataset_name = dataset_name
        
        if gt_dir is not None:
            self.gt_dir = gt_dir
            self.gt_files = sorted(os.listdir(self.gt_dir)) # NOTE: I assume that gt and img will have the same filename, so that this sort could return lists with same order.
        
        net_h, net_w = network_process_size[0], network_process_size[1]
        # self.resize = Resize(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
        if resize_mode == 'zoe':
            self.resize = Resize(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
            self.normalize = None
        elif resize_mode == 'depth-anything':
            self.resize = ResizeDA(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
            self.normalize = None
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.gt_dir is not None:
            image, depth = Images(self.rgb_image_dir, self.files, index, self.dataset_name), DepthMap(self.gt_dir, self.gt_files, index, self.dataset_name)
            image_rgb = image.rgb_image
            filename = image.name
            depth_gt = depth.gt
            edge = depth.edge
            
        else:
            image = Images(self.rgb_image_dir, self.files, index, self.dataset_name)
            image_rgb = image.rgb_image
            filename = image.name
        
        image_rgb = to_tensor(image_rgb).float()
        image_lr_tensor = self.resize(image_rgb.unsqueeze(dim=0)).squeeze(dim=0) # feed into deep branch (lr, zoe)
        
        if self.gt_dir is not None:
            boundary_tensor = to_tensor(edge)
            
            return_dict = \
                    {'image_lr': image_lr_tensor, 
                     'image_hr': image_rgb, 
                     'depth_gt': depth_gt, 
                     'boundary': boundary_tensor, 
                     'img_file_basename': filename}
        else:
            return_dict = \
                    {'image_lr': image_lr_tensor, 
                     'image_hr': image_rgb, 
                     'img_file_basename': filename}
        
        return return_dict
        
    def get_metrics(self, depth_gt, result, disp_gt_edges, **kwargs):
        return compute_metrics(\
            depth_gt, 
            result, 
            disp_gt_edges=disp_gt_edges, 
            min_depth_eval=self.min_depth, 
            max_depth_eval=self.max_depth, 
            garg_crop=False, 
            eigen_crop=False, 
            dataset=self.dataset_name)
    