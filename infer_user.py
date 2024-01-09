# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li

import os
import cv2
import argparse
from zoedepth.utils.config import get_config_user
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
import numpy as np
from zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from zoedepth.utils.misc import get_boundaries
from zoedepth.utils.misc import compute_metrics, RunningAverageDict
from tqdm import tqdm
import matplotlib
import torch.nn.functional as F
from zoedepth.data.middleburry import readPFM
import random
import imageio
from PIL import Image

def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state, strict=True)
    # model.load_state_dict(state, strict=False)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)

def load_ckpt(model, checkpoint):
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model


#### def dataset
def read_image(path, dataset_name):
    if dataset_name == 'u4k':
        img = np.fromfile(open(path, 'rb'), dtype=np.uint8).reshape(2160, 3840, 3) / 255.0
        img = img.astype(np.float32)[:, :, ::-1].copy()
    elif dataset_name == 'mid':
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), IMG_RESOLUTION, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0)
    
    elif dataset_name == 'nyu':
        img = Image.open(path)
        img = np.asarray(img, dtype=np.float32) / 255.0
        
    else:
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        print(img.shape)
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), IMG_RESOLUTION, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0)
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
            
            height = 1840
            width = 2300
            
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
        
        elif dataset_name == 'nyu':
            if pred:
                depth_gt = np.load(gt_path.replace('png', 'npy'))
                depth_gt = nn.functional.interpolate(
                    torch.tensor(depth_gt).unsqueeze(dim=0).unsqueeze(dim=0), (480, 640), mode='bilinear', align_corners=True).squeeze().numpy()
                
                edges = get_boundaries(depth_gt, th=1, dilation=0)
            else:
                depth_gt = np.asarray(Image.open(gt_path), dtype=np.float32) / 1000
                edges = get_boundaries(depth_gt, th=1, dilation=0)
            self.gt = depth_gt
            self.edge = edges
            
            
        else:
            raise NotImplementedError
        
        name = name.replace(".npy", "") # u4k
        name = name.replace(".exr", "") # gta
        self.name = name

class ImageDataset:
    def __init__(self, rgb_image_dir, gt_dir=None, dataset_name=''):
        self.rgb_image_dir = rgb_image_dir
        self.files = sorted(os.listdir(self.rgb_image_dir))
        self.gt_dir = gt_dir
        self.dataset_name = dataset_name

        if gt_dir is not None:
            self.gt_dir = gt_dir
            self.gt_files = sorted(os.listdir(self.gt_dir))
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.dataset_name == 'nyu':
            return Images(self.rgb_image_dir, self.files, index, self.dataset_name), DepthMap(self.gt_dir, self.gt_files, index, self.dataset_name), DepthMap('/ibex/ai/home/liz0l/codes/ZoeDepth/nfs/save/nyu', self.gt_files, index, self.dataset_name, pred=True)
        if self.gt_dir is not None:
            return Images(self.rgb_image_dir, self.files, index, self.dataset_name), DepthMap(self.gt_dir, self.gt_files, index, self.dataset_name)
        else:
            return Images(self.rgb_image_dir, self.files, index)

def crop(img, crop_bbox):
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    templete = torch.zeros((1, 1, img.shape[-2], img.shape[-1]), dtype=torch.float)
    templete[:, :, crop_y1:crop_y2, crop_x1:crop_x2] = 1.0
    img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    return img, templete

# def generatemask(size):
#     # Generates a Guassian mask
#     mask = np.zeros(size, dtype=np.float32)
#     sigma = int(size[0]/16)
#     k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
#     mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
#     mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
#     mask = (mask - mask.min()) / (mask.max() - mask.min())
#     mask = mask.astype(np.float32)
#     return mask

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.1*size[0]):size[0] - int(0.1*size[0]), int(0.1*size[1]): size[1] - int(0.1*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask


def generatemask_coarse(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/64)
    k_size = int(2 * np.ceil(2 * int(size[0]/64)) + 1)
    mask[int(0.001*size[0]):size[0] - int(0.001*size[0]), int(0.001*size[1]): size[1] - int(0.001*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

class RunningAverageMap:
    """A dictionary of running averages."""
    def __init__(self, average_map, count_map):
        self.average_map = average_map
        self.count_map = count_map
        self.average_map = self.average_map / self.count_map

    def update(self, pred_map, ct_map):
        self.average_map = (pred_map + self.count_map * self.average_map) / (self.count_map + ct_map)
        self.count_map = self.count_map + ct_map

# default size [540, 960]
# x_start, y_start = [0, 540, 1080, 1620], [0, 960, 1920, 2880]
def regular_tile(model, image, offset_x=0, offset_y=0, img_lr=None, iter_pred=None, boundary=0, update=False, avg_depth_map=None, blr_mask=False):
    # crop size
    # height = 540
    # width = 960
    height = CROP_SIZE[0]
    width = CROP_SIZE[1]

    assert offset_x >= 0 and offset_y >= 0
    
    tile_num_x = (IMG_RESOLUTION[1] - offset_x) // width
    tile_num_y = (IMG_RESOLUTION[0] - offset_y) // height
    x_start = [width * x + offset_x for x in range(tile_num_x)]
    y_start = [height * y + offset_y for y in range(tile_num_y)]
    imgs_crop = []
    crop_areas = []
    bboxs_roi = []
    bboxs_raw = []

    if iter_pred is not None:
        iter_pred = iter_pred.unsqueeze(dim=0).unsqueeze(dim=0)

    iter_priors = []
    for x in x_start: # w
        for y in y_start: # h
            bbox = (int(y), int(y+height), int(x), int(x+width))
            img_crop, crop_area = crop(image, bbox)
            imgs_crop.append(img_crop)
            crop_areas.append(crop_area)
            crop_y1, crop_y2, crop_x1, crop_x2 = bbox
            bbox_roi = torch.tensor([crop_x1 / IMG_RESOLUTION[1] * 512, crop_y1 / IMG_RESOLUTION[0] * 384, crop_x2 / IMG_RESOLUTION[1] * 512, crop_y2 / IMG_RESOLUTION[0] * 384])
            bboxs_roi.append(bbox_roi)
            bbox_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
            bboxs_raw.append(bbox_raw)

            if iter_pred is not None:
                iter_prior, _ = crop(iter_pred, bbox)
                iter_priors.append(iter_prior)

    crop_areas = torch.cat(crop_areas, dim=0)
    imgs_crop = torch.cat(imgs_crop, dim=0)
    bboxs_roi = torch.stack(bboxs_roi, dim=0)
    bboxs_raw = torch.stack(bboxs_raw, dim=0)

    if iter_pred is not None:
        iter_priors = torch.cat(iter_priors, dim=0)
        iter_priors = TRANSFORM(iter_priors)
        iter_priors = iter_priors.cuda().float()

    crop_areas = TRANSFORM(crop_areas)
    imgs_crop = TRANSFORM(imgs_crop)

    imgs_crop = imgs_crop.cuda().float()
    bboxs_roi = bboxs_roi.cuda().float()
    crop_areas = crop_areas.cuda().float()
    img_lr = img_lr.cuda().float()
    
    pred_depth_crops = []
    with torch.no_grad():
        for i, (img, bbox, crop_area) in enumerate(zip(imgs_crop, bboxs_roi, crop_areas)):

            if iter_pred is not None:
                iter_prior = iter_priors[i].unsqueeze(dim=0)
            else:
                iter_prior = None

            if i == 0:
                out_dict = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)
                whole_depth_pred = out_dict['coarse_depth_pred']
                # return whole_depth_pred.squeeze()
                # pred_depth_crop = out_dict['fine_depth_pred']
                pred_depth_crop = out_dict['metric_depth']
            else:
                pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['metric_depth']
                # pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['fine_depth_pred']


            pred_depth_crop = nn.functional.interpolate(
                pred_depth_crop, (height, width), mode='bilinear', align_corners=True)
            # pred_depth_crop = nn.functional.interpolate(
            #     pred_depth_crop, (height, width), mode='nearest')
            pred_depth_crops.append(pred_depth_crop.squeeze())

    whole_depth_pred = whole_depth_pred.squeeze()
    whole_depth_pred = nn.functional.interpolate(whole_depth_pred.unsqueeze(dim=0).unsqueeze(dim=0), IMG_RESOLUTION, mode='bilinear', align_corners=True).squeeze()

    ####### stich part
    inner_idx = 0
    init_flag = False
    if offset_x == 0 and offset_y == 0:
        init_flag = True
        # pred_depth = whole_depth_pred
        pred_depth = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
    else:
        iter_pred = iter_pred.squeeze()
        pred_depth = iter_pred

    blur_mask = generatemask((height, width)) + 1e-3
    count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)

    for ii, x in enumerate(x_start):
        for jj, y in enumerate(y_start):
            if init_flag:
                # pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx] + (1 - blur_mask) * crop_temp
                # pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx] + (1 - blur_mask) * crop_temp
                blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                count_map[y: y+height, x: x+width] = blur_mask
                pred_depth[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask

            else:
                # ensemble with running mean
                if blr_mask:
                    blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                    count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                    count_map[y: y+height, x: x+width] = blur_mask
                    pred_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                    pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask
                    avg_depth_map.update(pred_map, count_map)
                else:
                    if boundary != 0:
                        count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        count_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = 1
                        pred_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        pred_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = pred_depth_crops[inner_idx][boundary:-boundary, boundary:-boundary] 
                        avg_depth_map.update(pred_map, count_map)
                    else:
                        count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        count_map[y: y+height, x: x+width] = 1
                        pred_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx]
                        avg_depth_map.update(pred_map, count_map)


            inner_idx += 1

    if init_flag:
        avg_depth_map = RunningAverageMap(pred_depth, count_map)
        # blur_mask = generatemask_coarse(IMG_RESOLUTION)
        # blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
        # count_map = (1 - blur_mask)
        # pred_map = whole_depth_pred * (1 - blur_mask)
        # avg_depth_map.update(pred_map, count_map)
        return avg_depth_map

def regular_tile_param(model, image, offset_x=0, offset_y=0, img_lr=None, iter_pred=None, boundary=0, update=False, avg_depth_map=None, blr_mask=False, crop_size=None,
    img_resolution=None, transform=None):
    # crop size
    # height = 540
    # width = 960
    height = crop_size[0]
    width = crop_size[1]

    assert offset_x >= 0 and offset_y >= 0
    
    tile_num_x = (img_resolution[1] - offset_x) // width
    tile_num_y = (img_resolution[0] - offset_y) // height
    x_start = [width * x + offset_x for x in range(tile_num_x)]
    y_start = [height * y + offset_y for y in range(tile_num_y)]
    imgs_crop = []
    crop_areas = []
    bboxs_roi = []
    bboxs_raw = []

    if iter_pred is not None:
        iter_pred = iter_pred.unsqueeze(dim=0).unsqueeze(dim=0)

    iter_priors = []
    for x in x_start: # w
        for y in y_start: # h
            bbox = (int(y), int(y+height), int(x), int(x+width))
            img_crop, crop_area = crop(image, bbox)
            imgs_crop.append(img_crop)
            crop_areas.append(crop_area)
            crop_y1, crop_y2, crop_x1, crop_x2 = bbox
            bbox_roi = torch.tensor([crop_x1 / img_resolution[1] * 512, crop_y1 / img_resolution[0] * 384, crop_x2 / img_resolution[1] * 512, crop_y2 / img_resolution[0] * 384])
            bboxs_roi.append(bbox_roi)
            bbox_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
            bboxs_raw.append(bbox_raw)

            if iter_pred is not None:
                iter_prior, _ = crop(iter_pred, bbox)
                iter_priors.append(iter_prior)

    crop_areas = torch.cat(crop_areas, dim=0)
    imgs_crop = torch.cat(imgs_crop, dim=0)
    bboxs_roi = torch.stack(bboxs_roi, dim=0)
    bboxs_raw = torch.stack(bboxs_raw, dim=0)

    if iter_pred is not None:
        iter_priors = torch.cat(iter_priors, dim=0)
        iter_priors = transform(iter_priors)
        iter_priors = iter_priors.cuda().float()

    crop_areas = transform(crop_areas)
    imgs_crop = transform(imgs_crop)

    imgs_crop = imgs_crop.cuda().float()
    bboxs_roi = bboxs_roi.cuda().float()
    crop_areas = crop_areas.cuda().float()
    img_lr = img_lr.cuda().float()
    
    pred_depth_crops = []
    with torch.no_grad():
        for i, (img, bbox, crop_area) in enumerate(zip(imgs_crop, bboxs_roi, crop_areas)):

            if iter_pred is not None:
                iter_prior = iter_priors[i].unsqueeze(dim=0)
            else:
                iter_prior = None

            if i == 0:
                out_dict = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)
                whole_depth_pred = out_dict['coarse_depth_pred']
                # return whole_depth_pred.squeeze()
                # pred_depth_crop = out_dict['fine_depth_pred']
                pred_depth_crop = out_dict['metric_depth']
            else:
                pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['metric_depth']
                # pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['fine_depth_pred']


            pred_depth_crop = nn.functional.interpolate(
                pred_depth_crop, (height, width), mode='bilinear', align_corners=True)
            # pred_depth_crop = nn.functional.interpolate(
            #     pred_depth_crop, (height, width), mode='nearest')
            pred_depth_crops.append(pred_depth_crop.squeeze())

    whole_depth_pred = whole_depth_pred.squeeze()
    whole_depth_pred = nn.functional.interpolate(whole_depth_pred.unsqueeze(dim=0).unsqueeze(dim=0), img_resolution, mode='bilinear', align_corners=True).squeeze()

    ####### stich part
    inner_idx = 0
    init_flag = False
    if offset_x == 0 and offset_y == 0:
        init_flag = True
        # pred_depth = whole_depth_pred
        pred_depth = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
    else:
        iter_pred = iter_pred.squeeze()
        pred_depth = iter_pred

    blur_mask = generatemask((height, width)) + 1e-3
    count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)

    for ii, x in enumerate(x_start):
        for jj, y in enumerate(y_start):
            if init_flag:
                # pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx] + (1 - blur_mask) * crop_temp
                # pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx] + (1 - blur_mask) * crop_temp
                blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                count_map[y: y+height, x: x+width] = blur_mask
                pred_depth[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask

            else:
                # ensemble with running mean
                if blr_mask:
                    blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                    count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    count_map[y: y+height, x: x+width] = blur_mask
                    pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask
                    avg_depth_map.update(pred_map, count_map)
                else:
                    if boundary != 0:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = pred_depth_crops[inner_idx][boundary:-boundary, boundary:-boundary] 
                        avg_depth_map.update(pred_map, count_map)
                    else:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y: y+height, x: x+width] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx]
                        avg_depth_map.update(pred_map, count_map)


            inner_idx += 1

    if init_flag:
        avg_depth_map = RunningAverageMap(pred_depth, count_map)
        # blur_mask = generatemask_coarse(img_resolution)
        # blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
        # count_map = (1 - blur_mask)
        # pred_map = whole_depth_pred * (1 - blur_mask)
        # avg_depth_map.update(pred_map, count_map)
        return avg_depth_map

def random_tile(model, image, img_lr=None, iter_pred=None, boundary=0, update=False, avg_depth_map=None, blr_mask=False):
    height = CROP_SIZE[0]
    width = CROP_SIZE[1]
    
    
    x_start = [random.randint(0, IMG_RESOLUTION[1] - width - 1)]
    y_start = [random.randint(0, IMG_RESOLUTION[0] - height - 1)]
    
    imgs_crop = []
    crop_areas = []
    bboxs_roi = []
    bboxs_raw = []

    if iter_pred is not None:
        iter_pred = iter_pred.unsqueeze(dim=0).unsqueeze(dim=0)

    iter_priors = []
    for x in x_start: # w
        for y in y_start: # h
            bbox = (int(y), int(y+height), int(x), int(x+width))
            img_crop, crop_area = crop(image, bbox)
            imgs_crop.append(img_crop)
            crop_areas.append(crop_area)
            crop_y1, crop_y2, crop_x1, crop_x2 = bbox
            bbox_roi = torch.tensor([crop_x1 / IMG_RESOLUTION[1] * 512, crop_y1 / IMG_RESOLUTION[0] * 384, crop_x2 / IMG_RESOLUTION[1] * 512, crop_y2 / IMG_RESOLUTION[0] * 384])
            bboxs_roi.append(bbox_roi)
            bbox_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
            bboxs_raw.append(bbox_raw)

            if iter_pred is not None:
                iter_prior, _ = crop(iter_pred, bbox)
                iter_priors.append(iter_prior)

    crop_areas = torch.cat(crop_areas, dim=0)
    imgs_crop = torch.cat(imgs_crop, dim=0)
    bboxs_roi = torch.stack(bboxs_roi, dim=0)
    bboxs_raw = torch.stack(bboxs_raw, dim=0)

    if iter_pred is not None:
        iter_priors = torch.cat(iter_priors, dim=0)
        iter_priors = TRANSFORM(iter_priors)
        iter_priors = iter_priors.cuda().float()

    crop_areas = TRANSFORM(crop_areas)
    imgs_crop = TRANSFORM(imgs_crop)
    


    imgs_crop = imgs_crop.cuda().float()
    bboxs_roi = bboxs_roi.cuda().float()
    crop_areas = crop_areas.cuda().float()
    img_lr = img_lr.cuda().float()
    
    pred_depth_crops = []
    with torch.no_grad():
        for i, (img, bbox, crop_area) in enumerate(zip(imgs_crop, bboxs_roi, crop_areas)):

            if iter_pred is not None:
                iter_prior = iter_priors[i].unsqueeze(dim=0)
            else:
                iter_prior = None

            if i == 0:
                out_dict = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)
                whole_depth_pred = out_dict['coarse_depth_pred']
                pred_depth_crop = out_dict['metric_depth']
                # return whole_depth_pred.squeeze()
            else:
                pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['metric_depth']


            pred_depth_crop = nn.functional.interpolate(
                pred_depth_crop, (height, width), mode='bilinear', align_corners=True)
            # pred_depth_crop = nn.functional.interpolate(
            #     pred_depth_crop, (height, width), mode='nearest')
            pred_depth_crops.append(pred_depth_crop.squeeze())

    whole_depth_pred = whole_depth_pred.squeeze()

    ####### stich part
    inner_idx = 0
    init_flag = False
    iter_pred = iter_pred.squeeze()
    pred_depth = iter_pred

    blur_mask = generatemask((height, width)) + 1e-3
    for ii, x in enumerate(x_start):
        for jj, y in enumerate(y_start):
            if init_flag:
                # wont be here
                crop_temp = copy.deepcopy(whole_depth_pred[y: y+height, x: x+width])
                blur_mask = torch.ones((height, width))
                blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx]+ (1 - blur_mask) * crop_temp
            else:
                if blr_mask:
                    blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                    count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                    count_map[y: y+height, x: x+width] = blur_mask
                    pred_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                    pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask
                    avg_depth_map.update(pred_map, count_map)
                else:
                    # ensemble with running mean
                    if boundary != 0:
                        count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        count_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = 1
                        pred_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        pred_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = pred_depth_crops[inner_idx][boundary:-boundary, boundary:-boundary] 
                        avg_depth_map.update(pred_map, count_map)
                    else:
                        count_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        count_map[y: y+height, x: x+width] = 1
                        pred_map = torch.zeros(IMG_RESOLUTION, device=pred_depth_crops[inner_idx].device)
                        pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx]
                        avg_depth_map.update(pred_map, count_map)

            inner_idx += 1

    if avg_depth_map is None:
        return pred_depth


def random_tile_param(model, image, img_lr=None, iter_pred=None, boundary=0, update=False, avg_depth_map=None, blr_mask=False, crop_size=None,
    img_resolution=None, transform=None):
    height = crop_size[0]
    width = crop_size[1]
    
    
    x_start = [random.randint(0, img_resolution[1] - width - 1)]
    y_start = [random.randint(0, img_resolution[0] - height - 1)]
    
    imgs_crop = []
    crop_areas = []
    bboxs_roi = []
    bboxs_raw = []

    if iter_pred is not None:
        iter_pred = iter_pred.unsqueeze(dim=0).unsqueeze(dim=0)

    iter_priors = []
    for x in x_start: # w
        for y in y_start: # h
            bbox = (int(y), int(y+height), int(x), int(x+width))
            img_crop, crop_area = crop(image, bbox)
            imgs_crop.append(img_crop)
            crop_areas.append(crop_area)
            crop_y1, crop_y2, crop_x1, crop_x2 = bbox
            bbox_roi = torch.tensor([crop_x1 / img_resolution[1] * 512, crop_y1 / img_resolution[0] * 384, crop_x2 / img_resolution[1] * 512, crop_y2 / img_resolution[0] * 384])
            bboxs_roi.append(bbox_roi)
            bbox_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
            bboxs_raw.append(bbox_raw)

            if iter_pred is not None:
                iter_prior, _ = crop(iter_pred, bbox)
                iter_priors.append(iter_prior)

    crop_areas = torch.cat(crop_areas, dim=0)
    imgs_crop = torch.cat(imgs_crop, dim=0)
    bboxs_roi = torch.stack(bboxs_roi, dim=0)
    bboxs_raw = torch.stack(bboxs_raw, dim=0)

    if iter_pred is not None:
        iter_priors = torch.cat(iter_priors, dim=0)
        iter_priors = transform(iter_priors)
        iter_priors = iter_priors.cuda().float()

    crop_areas = transform(crop_areas)
    imgs_crop = transform(imgs_crop)
    
    imgs_crop = imgs_crop.cuda().float()
    bboxs_roi = bboxs_roi.cuda().float()
    crop_areas = crop_areas.cuda().float()
    img_lr = img_lr.cuda().float()
    
    pred_depth_crops = []
    with torch.no_grad():
        for i, (img, bbox, crop_area) in enumerate(zip(imgs_crop, bboxs_roi, crop_areas)):

            if iter_pred is not None:
                iter_prior = iter_priors[i].unsqueeze(dim=0)
            else:
                iter_prior = None

            if i == 0:
                out_dict = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)
                whole_depth_pred = out_dict['coarse_depth_pred']
                pred_depth_crop = out_dict['metric_depth']
                # return whole_depth_pred.squeeze()
            else:
                pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['metric_depth']


            pred_depth_crop = nn.functional.interpolate(
                pred_depth_crop, (height, width), mode='bilinear', align_corners=True)
            # pred_depth_crop = nn.functional.interpolate(
            #     pred_depth_crop, (height, width), mode='nearest')
            pred_depth_crops.append(pred_depth_crop.squeeze())

    whole_depth_pred = whole_depth_pred.squeeze()

    ####### stich part
    inner_idx = 0
    init_flag = False
    iter_pred = iter_pred.squeeze()
    pred_depth = iter_pred

    blur_mask = generatemask((height, width)) + 1e-3
    for ii, x in enumerate(x_start):
        for jj, y in enumerate(y_start):
            if init_flag:
                # wont be here
                crop_temp = copy.deepcopy(whole_depth_pred[y: y+height, x: x+width])
                blur_mask = torch.ones((height, width))
                blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx]+ (1 - blur_mask) * crop_temp
            else:

                if blr_mask:
                    blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
                    count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    count_map[y: y+height, x: x+width] = blur_mask
                    pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask
                    avg_depth_map.update(pred_map, count_map)
                else:
                    # ensemble with running mean
                    if boundary != 0:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = pred_depth_crops[inner_idx][boundary:-boundary, boundary:-boundary] 
                        avg_depth_map.update(pred_map, count_map)
                    else:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y: y+height, x: x+width] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx]
                        avg_depth_map.update(pred_map, count_map)

            inner_idx += 1

    if avg_depth_map is None:
        return pred_depth


def colorize_infer(value, cmap='magma_r', vmin=None, vmax=None):
    # normalize
    vmin = value.min() if vmin is None else vmin
    # vmax = value.max() if vmax is None else vmax
    vmax = np.percentile(value, 95) if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :3] # bgr -> rgb
    rgb_value = value[..., ::-1]

    return rgb_value


def colorize(value, vmin=None, vmax=None, cmap='turbo_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None, dataset_name=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    # vmin = np.percentile(value[mask],2) if vmin is None else vmin
    # vmin = value.min() if vmin is None else vmin
    # vmax = np.percentile(value[mask],95) if vmax is None else vmax
    
    # mid gt
    if dataset_name == 'mid':
        vmin = np.percentile(value[mask],2) if vmin is None else vmin
        vmax = np.percentile(value[mask],85) if vmax is None else vmax
    else:
        vmin = value.min() if vmin is None else vmin
        vmax = np.percentile(value[mask],95) if vmax is None else vmax
        
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def rescale(A, lbound=0, ubound=1):
    """
    Rescale an array to [lbound, ubound].

    Parameters:
    - A: Input data as numpy array
    - lbound: Lower bound of the scale, default is 0.
    - ubound: Upper bound of the scale, default is 1.

    Returns:
    - Rescaled array
    """
    A_min = np.min(A)
    A_max = np.max(A)
    return (ubound - lbound) * (A - A_min) / (A_max - A_min) + lbound

def run(model, dataset, gt_dir=None, show_path=None, show=False, save_flag=False, save_path=None, mode=None, dataset_name=None, base_zoed=False, blr_mask=False):
    data_len = len(dataset)

    if gt_dir is not None:
        metrics_avg = RunningAverageDict()

    for image_ind in tqdm(range(data_len)):
        if dataset_name == 'nyu':
            images, depths, pred_depths = dataset[image_ind]
        else:
            if gt_dir is None:
                images = dataset[image_ind]
            else:
                images, depths = dataset[image_ind]

        # Load image from dataset
        img = torch.tensor(images.rgb_image).unsqueeze(dim=0).permute(0, 3, 1, 2) # shape: 1, 3, h, w
        img_lr = TRANSFORM(img)
        
        
        if base_zoed:
            with torch.no_grad():
                pred_depth = model(img.cuda())['metric_depth'].squeeze()
            avg_depth_map = RunningAverageMap(pred_depth)
            
        else:
            # pred_depth, count_map = regular_tile(model, img, offset_x=0, offset_y=0, img_lr=img_lr)
            # avg_depth_map = RunningAverageMap(pred_depth, count_map)
            avg_depth_map = regular_tile(model, img, offset_x=0, offset_y=0, img_lr=img_lr)
        
            if mode== 'p16':
                pass
            elif mode== 'p49':
                regular_tile(model, img, offset_x=CROP_SIZE[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
                regular_tile(model, img, offset_x=0, offset_y=CROP_SIZE[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
                regular_tile(model, img, offset_x=CROP_SIZE[1]//2, offset_y=CROP_SIZE[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

            elif mode[0] == 'r':
                regular_tile(model, img, offset_x=CROP_SIZE[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
                regular_tile(model, img, offset_x=0, offset_y=CROP_SIZE[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
                regular_tile(model, img, offset_x=CROP_SIZE[1]//2, offset_y=CROP_SIZE[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

                for i in tqdm(range(int(mode[1:]))):
                    random_tile(model, img, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=BOUNDARY, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
                
        if show:
            color_map = copy.deepcopy(avg_depth_map.average_map)
            color_map_1 = colorize_infer(color_map.detach().cpu().numpy())
            cv2.imwrite(os.path.join(show_path, '{}.png'.format(images.name)), color_map_1)

            color_map = copy.deepcopy(avg_depth_map.average_map)
            color_map_2 = colorize(color_map, cmap='gray_r')
            cv2.imwrite(os.path.join(show_path, '{}_gray.png'.format(images.name)), color_map_2)


        if save_flag:
            np.save(os.path.join(save_path, '{}.npy'.format(images.name)), avg_depth_map.average_map.squeeze().detach().cpu().numpy())


            png_file = Image.fromarray((avg_depth_map.average_map.squeeze().detach().cpu().numpy()*256).astype('uint16'))
            png_file.save(os.path.join(show_path, '{}_16bitpng.png'.format(images.name)))
            # np.save(os.path.join(save_path, '{}.npy'.format(images.name)), depths.gt)
            
        if gt_dir is not None:
            if dataset_name == 'nyu':
                metrics = compute_metrics(torch.tensor(depths.gt), avg_depth_map.average_map, disp_gt_edges=depths.edge, min_depth_eval=1e-3, max_depth_eval=10, garg_crop=False, eigen_crop=True, dataset='nyu', pred_depths=torch.tensor(pred_depths.gt))
                # metrics = compute_metrics(torch.tensor(depths.gt), avg_depth_map.average_map, disp_gt_edges=depths.edge, min_depth_eval=1e-3, max_depth_eval=10, garg_crop=False, eigen_crop=True, dataset='nyu')
            else:
                metrics = compute_metrics(torch.tensor(depths.gt), avg_depth_map.average_map, disp_gt_edges=depths.edge, min_depth_eval=1e-3, max_depth_eval=80, garg_crop=False, eigen_crop=False, dataset='')
            metrics_avg.update(metrics)
            print(metrics)
            
    if gt_dir is not None:
        print(metrics_avg.get_value())
    else:
        print("successful!")
    return avg_depth_map

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, required=True)
    parser.add_argument('--show_path', type=str, required=None)
    parser.add_argument("--ckp_path", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, default="zoedepth")
    parser.add_argument("--model_cfg_path", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--img_resolution", type=str, default=None)
    parser.add_argument("--crop_size", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--base_zoed", action='store_true')
    parser.add_argument("--boundary", type=int, default=0)
    parser.add_argument("--blur_mask", action='store_true')
    args, unknown_args = parser.parse_known_args()

    # prepare some global args
    global IMG_RESOLUTION
    if args.dataset_name == 'u4k':
        IMG_RESOLUTION = (2160, 3840)
    elif args.dataset_name == 'gta':
        IMG_RESOLUTION = (1080, 1920)
    elif args.dataset_name == 'nyu':
        IMG_RESOLUTION = (480, 640)
    else:
        IMG_RESOLUTION = (2160, 3840)
    
    global TRANSFORM 
    TRANSFORM = Compose([Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")])
    global BOUNDARY 
    BOUNDARY = args.boundary
    
    if args.img_resolution is not None:
        IMG_RESOLUTION = (int(args.img_resolution.split('x')[0]), int(args.img_resolution.split('x')[1]))
    global CROP_SIZE 
    CROP_SIZE = (int(IMG_RESOLUTION[0] // 4), int(IMG_RESOLUTION[1] // 4))
    if args.crop_size is not None:
        CROP_SIZE = (int(args.crop_size.split('x')[0]), int(args.crop_size.split('x')[1]))
    print("\nCurrent image resolution: {}\n Current crop size: {}".format(IMG_RESOLUTION, CROP_SIZE))
    
    overwrite_kwargs = parse_unknown(unknown_args)
    overwrite_kwargs['model_cfg_path'] = args.model_cfg_path
    overwrite_kwargs["model"] = args.model
        
    # blur_mask_crop = generatemask(CROP_SIZE)
    # plt.imshow(blur_mask_crop)
    # plt.savefig('./nfs/results_show/crop_mask.png')
    # blur_mask_crop = generatemask_coarse(IMG_RESOLUTION)
    # plt.imshow(blur_mask_crop)
    # plt.savefig('./nfs/results_show/whole_mask.png')

    config = get_config_user(args.model, **overwrite_kwargs)
    config["pretrained_resource"] = ''
    model = build_model(config)
    model = load_ckpt(model, args.ckp_path)
    model.eval()
    model.cuda()

    # Create dataset from input images
    dataset_custom = ImageDataset(args.rgb_dir, args.gt_dir, args.dataset_name)

    # start running
    if args.show:
        os.makedirs(args.show_path, exist_ok=True)
    if args.save:
        os.makedirs(args.save_path, exist_ok=True)
        
    run(model, dataset_custom, args.gt_dir, args.show_path, args.show, args.save, args.save_path, mode=args.mode, dataset_name=args.dataset_name, base_zoed=args.base_zoed, blr_mask=args.blur_mask)
