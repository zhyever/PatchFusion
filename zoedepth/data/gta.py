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

# This file is partly inspired from ZoeDepth (https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/data/data_mono.py); author: Shariq Farooq Bhat

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os.path as osp
import random
import torch.nn as nn
import cv2
import copy
from zoedepth.utils.misc import get_boundaries
from zoedepth.models.base_models.midas import Resize


from .u4k import U4KDataset, remove_leading_slash

import re
import numpy as np
import sys
import matplotlib.pyplot as plt
import imageio

class GTA(U4KDataset):
    def __init__(self, config, mode, data_root, split):
        super().__init__(config, mode, data_root, split)
        self.crop_size = [270, 480] # 1/4

    def load_data_list(self):
        # os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        
        """Load annotation from directory.
        Args:
            data_root (str): Data root for img_dir/ann_dir.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        data_root = self.data_root
        split = self.split
        
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info_l = dict()

                    img_l, depth_map_l = line.strip().split(" ")

                    img_info_l['depth_map_path'] = osp.join(data_root, remove_leading_slash(depth_map_l))
                    img_info_l['img_path'] = osp.join(data_root, remove_leading_slash(img_l))
                    img_info_l['depth_fields'] = []

                    img_infos.append(img_info_l)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        return img_infos

    
    def __getitem__(self, idx):

        img_file_path = self.data_infos[idx]['img_path']
        disp_path = self.data_infos[idx]['depth_map_path']

        height=1080 
        width=1920

        image = Image.open(img_file_path).convert("RGB")
        image = np.asarray(image, dtype=np.uint8) / 255.0
        image = image.astype(np.float32)

        # disp_gt, scale = readPFM(disp_path)
        
        # ref_depth_full = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        ref_depth_full = imageio.imread(disp_path)
        ref_depth_full = np.array(ref_depth_full).astype(np.float32) / 256
        # invalid_mask_full = np.logical_or(ref_depth_full < self.config.min_depth, ref_depth_full > self.config.max_depth)
        # ref_depth_full[invalid_mask_full] = 0
        # ref_depth_full[ref_depth_full < self.config.min_depth] = self.config.min_depth
        # ref_depth_full[ref_depth_full > self.config.max_depth] = self.config.max_depth

        depth_gt = ref_depth_full
        disp_gt_copy = depth_gt
        depth_gt = depth_gt[..., np.newaxis]
        depth_gt[depth_gt > self.config.max_depth] = self.config.max_depth # for vis

        # disp_gt_edges = get_boundaries(disp_gt_copy, th=1/20, dilation=0)
        
        bbox = None
        bboxs_res = None
        crop_areas = None
        bboxs_roi = None # hack for infer

        if self.mode == 'train':
            image, depth_gt = self.train_preprocess(image, depth_gt)
            img_temp = copy.deepcopy(image)
            depth_gt_temp = copy.deepcopy(depth_gt)
            
            if self.random_crop: # use in sec_stage
                if self.consistency_training:
                    crop_y1, crop_y2, crop_x1, crop_x2 = self.get_crop_bbox(image) # ensure the prob of crop is the same
                    while True:
                        # shift_x = random.randint(self.overlap_length//3, self.overlap_length)
                        # shift_y = random.randint(self.overlap_length//3, self.overlap_length)
                        shift_x = self.overlap_length_w
                        shift_y = self.overlap_length_h
                        if random.random() > 0.5:
                            shift_x = shift_x * -1
                        if random.random() > 0.5:
                            shift_y = shift_y * -1
                        crop_y1_shift, crop_y2_shift, crop_x1_shift, crop_x2_shift = crop_y1 + shift_y, crop_y2 + shift_y, crop_x1 + shift_x, crop_x2 + shift_x
                        if crop_y1_shift > 0 and crop_x1_shift > 0 and crop_y2_shift < image.shape[0] and crop_x2_shift < image.shape[1]:
                            break
                    bbox_ori = (crop_y1, crop_y2, crop_x1, crop_x2)
                    bbox_shift = (crop_y1_shift, crop_y2_shift, crop_x1_shift, crop_x2_shift)
                    image_ori, crop_area_ori = self.crop(image, bbox_ori, tmp=True)
                    image_shift, crop_area_shift = self.crop(image, bbox_shift, tmp=True)
                    depth_gt_ori = self.crop(depth_gt, bbox_ori)
                    depth_gt_shift = self.crop(depth_gt, bbox_shift)
                    disp_gt_copy_ori = self.crop(disp_gt_copy, bbox_ori)
                    disp_gt_copy_shift = self.crop(disp_gt_copy, bbox_shift)

                    bboxs_ori = torch.tensor([crop_x1 / width * 512, crop_y1 / height * 384, crop_x2 / width * 512, crop_y2 / height * 384])
                    bboxs_shift = torch.tensor([crop_x1_shift / width * 512, crop_y1_shift / height * 384, crop_x2_shift / width * 512, crop_y2_shift / height * 384])
                    bboxs_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
                    bboxs_raw_shift = torch.tensor([crop_x1_shift, crop_y1_shift, crop_x2_shift, crop_y2_shift]) 

                else:
                    bbox = self.get_crop_bbox(image)
                    image, crop_area = self.crop(image, bbox, tmp=True)
                    depth_gt = self.crop(depth_gt, bbox)
                    disp_gt_copy = self.crop(disp_gt_copy, bbox)

                    crop_y1, crop_y2, crop_x1, crop_x2 = bbox
                    bboxs_res = torch.tensor([crop_x1 / width * 512, crop_y1 / height * 384, crop_x2 / width * 512, crop_y2 / height * 384]) # coord in 384, 512
                    bboxs_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 


            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            mask_raw = np.logical_and(depth_gt_temp > self.config.min_depth, depth_gt_temp < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'depth': depth_gt, 'focal': 0, 'mask': mask, 'image_raw': image.copy(), 'mask_raw': mask_raw}
            
            
            if self.random_crop:
                if self.consistency_training:
                    image = np.concatenate([image_ori, image_shift], axis=-1)
                    depth_gt = np.concatenate([depth_gt_ori, depth_gt_shift], axis=-1)
                    crop_area = np.concatenate([crop_area_ori, crop_area_shift], axis=-1)
                    bboxs_res = torch.cat([bboxs_ori, bboxs_shift], dim=-1)
                    bboxes_raw_res = torch.cat([bboxs_raw, bboxs_raw_shift], dim=-1)
                    mask = np.logical_and(depth_gt > self.config.min_depth,
                                          depth_gt < self.config.max_depth)

                    # hack the sample dict
                    sample['image'] = image
                    sample['depth'] = depth_gt
                    sample['crop_area'] = crop_area
                    sample['bbox'] = bboxs_res
                    sample['bbox_raw'] = bboxes_raw_res
                    sample['shift'] = torch.tensor([shift_y, shift_x]) # h direction, then w direction
                    sample['mask'] = mask
                        
                else:
                    if bboxs_res is not None:
                        sample['bbox'] = bboxs_res
                        sample['bbox_raw'] = bboxs_raw
                    sample['crop_area'] = crop_area

            if self.sampled_training:
                self.data_sampler(sample, disp_gt_copy)

                # update mask
                sample_points = sample['sample_points']
                sample_mask = np.logical_and(sample_points[:, -1] > self.config.min_depth,
                                             sample_points[:, -1] < self.config.max_depth).squeeze()[None, ...]
                sample['sample_mask'] = sample_mask

            
        else:
            # nothing needs to be changed for consistency training.

            img_temp = copy.deepcopy(image)
            depth_gt_temp = copy.deepcopy(depth_gt)
            if self.sec_stage:
                # x_start, y_start = [0, 540, 1080, 1620], [0, 960, 1920, 2880]
                x_start, y_start = [0 + 3 * self.overlap / 2, 270 + self.overlap / 2, 540 - self.overlap / 2, 810 - 3 * self.overlap / 2], \
                    [0 + 3 * self.overlap / 2, 480 + self.overlap / 2, 960 - self.overlap / 2, 1440 - 3 * self.overlap / 2]
                img_crops = []
                bboxs_roi = []
                crop_areas = []
                bboxs_raw_list = []
                for x in x_start:
                    for y in y_start:
                        bbox = (int(x), int(x+270), int(y), int(y+480))
                        img_crop, crop_area = self.crop(image, bbox, tmp=True)
                        img_crops.append(img_crop)
                        crop_areas.append(crop_area)
                        crop_y1, crop_y2, crop_x1, crop_x2 = bbox
                        bbox_roi = torch.tensor([crop_x1 / width * 512, crop_y1 / height * 384, crop_x2 / width * 512, crop_y2 / height * 384])
                        bboxs_roi.append(bbox_roi)
                        bboxs_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
                        bboxs_raw_list.append(bboxs_raw)

                image = img_crops
                bboxs_roi = torch.stack(bboxs_roi, dim=0)
                bboxs_raw = torch.stack(bboxs_raw_list, dim=0)

                # bbox = (820, 1360 ,1440, 2400) # a hack version for quick evaluation
                # image = self.crop(image, bbox)
                # depth_gt = self.crop(depth_gt, bbox)
                # disp_gt_copy = self.crop(disp_gt_copy, bbox)

            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            
            disp_gt_edges = get_boundaries(disp_gt_copy, th=1, dilation=0)
            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': 0, 'has_valid_depth': True,
                          'image_path': img_file_path, 'depth_path': disp_path, 'depth_factor_path': 0,
                          'mask': mask, 'image_raw': image.copy(), 'disp_gt_edges': disp_gt_edges}
                if bboxs_roi is not None:
                    sample['bbox'] = bboxs_roi
                    sample['bbox_raw'] = bboxs_raw
                if crop_areas is not None:
                    sample['crop_area'] = crop_areas
                

            else:
                sample = {'image': image, 'focal': 0, 'image_raw': image.copy(), 'disp_gt_edges': disp_gt_edges, 'image_path': img_file_path}
                if bboxs_roi is not None:
                    sample['bbox'] = bboxs_roi
                    sample['bbox_raw'] = bboxs_raw
                if crop_areas is not None:
                    sample['crop_area'] = crop_areas
                
        if self.transform:
            sample['img_temp'] = img_temp
            sample['depth_gt_temp'] = depth_gt_temp
            sample = self.transform(sample)

        sample['dataset'] = self.config.dataset
        return sample

    def __len__(self):
        return len(self.data_infos)
    
def get_gta_loader(config, mode, transform):
    if mode == 'train':
        log = 0
        dataset = GTA(config, mode, config.data_path, config.filenames_train)
        dataset[0]
        if config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
        
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=(train_sampler is None),
                                num_workers=config.workers,
                                pin_memory=True,
                                persistent_workers=True,
                                sampler=train_sampler)
            
    elif mode == 'online_eval':
        dataset = GTA(config, mode, config.data_path, config.filenames_val)
        # dataset = U4KDataset(config, mode, config.data_path, config.filenames_train)


        if config.distributed:  # redundant. here only for readability and to be more explicit
            # Give whole test set to all processes (and report evaluation only on one) regardless
            eval_sampler = None
        else:
            eval_sampler = None

        dataloader = DataLoader(dataset, 1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=False,
                                sampler=eval_sampler)
        
    else:
        dataset = GTA(config, mode, config.data_path, config.filenames_test)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=1)
        
    return dataloader

