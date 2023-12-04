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

class SampleDataPairs(object):
    
    def __init__(self,
                 num_sample_inout=50000,
                 sampling_strategy='random', # or 'dda'
                 dilation_factor=10,
                 crop_size=(2160, 3840),
                 ):

        self.num_sample_inout = num_sample_inout
        self.sampling_strategy = sampling_strategy
        self.dilation_factor = dilation_factor
        self.crop_height, self.crop_width = crop_size[0], crop_size[1]
        self.__init_grid()

    def __init_grid(self):
        nu = np.linspace(0, self.crop_width - 1, self.crop_width)
        nv = np.linspace(0, self.crop_height - 1, self.crop_height)
        u, v = np.meshgrid(nu, nv)

        self.u = u.flatten()
        self.v = v.flatten()

    def get_coords(self, gt):
        #  get subpixel coordinates
        u = self.u + np.random.random_sample(self.u.size)
        v = self.v + np.random.random_sample(self.v.size)

        # use nearest neighbor to get gt for each samples
        d = gt[np.clip(np.rint(v).astype(np.uint16), 0, self.crop_height-1),
               np.clip(np.rint(u).astype(np.uint16), 0, self.crop_width-1)]

        # remove invalid depth values
        u = u[np.nonzero(d)]
        v = v[np.nonzero(d)]
        d = d[np.nonzero(d)]

        return np.stack((u, v, d), axis=-1)


    def get_boundaries(self, disp, th=1., dilation=10):
        edges_y = np.logical_or(np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
                                np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))))
        edges_x = np.logical_or(np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
                                np.pad(np.abs(disp[:, :-1] - disp[:,1:]) > th, ((0, 0), (0, 1))))
        edges = np.logical_or(edges_y,  edges_x).astype(np.float32)

        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        return edges


    def __call__(self, results, disp_gt_copy):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        gt = results['depth']
        gt_squeeze = gt[:, :, 0]
        if self.sampling_strategy == "random":
            random_points = self.get_coords(gt_squeeze)
            idx = np.random.choice(random_points.shape[0], self.num_sample_inout)
            points = random_points[idx, :]

        elif self.sampling_strategy == "dda":
            disp_gt_squeeze = disp_gt_copy
            edges = self.get_boundaries(disp_gt_squeeze, dilation=self.dilation_factor)
            random_points = self.get_coords(gt_squeeze * (1. - edges))
            edge_points = self.get_coords(gt_squeeze * edges)

            # if edge points exist
            if edge_points.shape[0]>0 and random_points.shape[0]>0:
                # Check tot num of edge points
                cond = edges.sum()//2 -  self.num_sample_inout//2 < 0
                tot= (self.num_sample_inout - int(edges.sum())//2, int(edges.sum())//2) if cond else \
                     (self.num_sample_inout//2, self.num_sample_inout//2)

                idx = np.random.choice(random_points.shape[0], tot[0])
                idx_edges = np.random.choice(edge_points.shape[0], tot[1])
                points = np.concatenate([edge_points[idx_edges, :], random_points[idx, :]], 0)
            # use uniform sample otherwise
            else:
                random_points = self.get_coords(gt_squeeze)
                idx = np.random.choice(random_points.shape[0], self.num_sample_inout)
                points = random_points[idx, :]

        
        results['sample_points'] = points

        return results

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ToTensor(object):
    def __init__(self, mode, do_normalize=False, size=None, sec_stage=False):
        self.mode = mode
        # don't do normalization as default
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            # self.resize = transforms.Resize(size=size)
            net_h, net_w = size
            self.resize = Resize(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
        else:
            self.resize = nn.Identity()
        self.sec_stage = sec_stage

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        crop_areas = sample.get('crop_area', None)

        if isinstance(image, list):
            # there must be crop_areas
            # only infer on eval sec_stage
            imgs_process = []
            crp_process = []
            for img, crp in zip(image, crop_areas):
                img = self.to_tensor(img)
                img = self.normalize(img)
                img = img.unsqueeze(dim=0)
                img = self.resize(img)
                img = img.squeeze(dim=0)
                imgs_process.append(img)

                crp = self.to_tensor(crp)
                crp = crp.unsqueeze(dim=0)
                crp = self.resize(crp)
                crp = crp.squeeze(dim=0)
                crp_process.append(crp)

            image = torch.cat(imgs_process, dim=0)
            crop_areas = torch.cat(crp_process, dim=0)

            img_temp = sample['img_temp']
            img_temp = self.to_tensor(img_temp)
            img_temp = self.normalize(img_temp)
            img_temp = img_temp.unsqueeze(dim=0)
            img_temp = self.resize(img_temp) #NOTE: hack
            img_temp = img_temp.squeeze(dim=0)
            image_raw = copy.deepcopy(img_temp)

            
        else:
            image = self.to_tensor(image)
            image = self.normalize(image)

            if crop_areas is not None:
                crop_areas = self.to_tensor(crop_areas)
                crop_areas = crop_areas.unsqueeze(dim=0)
                crop_areas = self.resize(crop_areas)
                crop_areas = crop_areas.squeeze(dim=0)

            if self.sec_stage:
                img_temp = sample['img_temp']
                img_temp = self.to_tensor(img_temp)
                img_temp = self.normalize(img_temp)

                img_temp = img_temp.unsqueeze(dim=0)
                img_temp = self.resize(img_temp)
                image_raw = img_temp.squeeze(dim=0)

                image = image.unsqueeze(dim=0)
                image = self.resize(image)
                image = image.squeeze(dim=0)

            else:
                # in the first stage, this hr info is reserved
                image_raw = copy.deepcopy(image)
                image = image.unsqueeze(dim=0)
                image = self.resize(image)
                image = image.squeeze(dim=0)

        if self.mode == 'test':
            return_dict =  {'image': image, 'focal': focal}
            if crop_areas is not None:
                return_dict['crop_area'] = crop_areas
            return return_dict
        

        depth = sample['depth']
        depth = self.to_tensor(depth)
        depth_gt_temp = sample['depth_gt_temp']
        depth_gt_raw = self.to_tensor(depth_gt_temp)
        
        if self.mode == 'train':
            return_dict = {**sample, 'image': image, 'depth': depth, 'focal': focal, 'image_raw': image_raw, 'depth_raw': depth_gt_raw}
            if crop_areas is not None:
                return_dict['crop_area'] = crop_areas
            return return_dict
        else:
            has_valid_depth = sample['has_valid_depth']
            # image = self.resize(image)
            return_dict = {**sample, 'image': image, 'depth': depth, 'focal': focal, 'image_raw': image_raw, 
                    'has_valid_depth': has_valid_depth, 'image_path': sample['image_path'], 'depth_path': sample['depth_path'],
                    'depth_raw': depth_gt_raw}
            if crop_areas is not None:
                return_dict['crop_area'] = crop_areas
            return return_dict

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1))) # img here
            return img


def preprocessing_transforms(mode, sec_stage=False, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, sec_stage=sec_stage, **kwargs)
    ])

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

class U4KDataset(Dataset):
    def __init__(self, config, mode, data_root, split):
        self.mode = mode
        self.config = config
        self.data_root = data_root
        self.split = split

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        self.sec_stage = self.config.get("sec_stage", False)
        self.transform = preprocessing_transforms(mode, size=img_size, sec_stage=self.sec_stage)

        self.data_infos = self.load_data_list()
        
        self.sampled_training = self.config.get("sampled_training", False)
        if self.sampled_training:
            self.data_sampler = SampleDataPairs(
                num_sample_inout=config.num_sample_inout,
                sampling_strategy=config.sampling_strategy, # or 'dda'
                dilation_factor=config.dilation_factor,
                crop_size=config.transform_sample_gt_size)
        
        self.random_crop = self.config.get("random_crop", False)
        self.crop_size = [540, 960] # 1/4
        self.overlap = self.config.get("overlap", False)

        self.consistency_training = self.config.get("consistency_training", False)
        self.overlap_length_h = self.config.get("overlap_length_h", int(256))
        self.overlap_length_w = self.config.get("overlap_length_w", int(384))
        print("current overlap_length_h and overlap_length_w are {} and {}".format(self.overlap_length_h, self.overlap_length_w))
        
        
    def load_data_list(self):
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
                    # img_info_r = dict()

                    img_l, img_r, depth_map_l, depth_map_r = line.strip().split(" ")

                    # HACK: a hack to replace the png with raw to accelerate training
                    img_l = img_l[:-3] + 'raw'
                    # img_r = img_r[:-3] + 'raw'

                    img_info_l['depth_map_path'] = osp.join(data_root, remove_leading_slash(depth_map_l))
                    # img_info_r['depth_map_path'] = osp.join(data_root, remove_leading_slash(depth_map_r))

                    img_info_l['img_path'] = osp.join(data_root, remove_leading_slash(img_l))
                    # img_info_r['filename'] = osp.join(data_root, remove_leading_slash(img_r))

                    img_info_l['depth_fields'] = []

                    filename = img_info_l['depth_map_path']
                    ext_name_l = filename.replace('Disp0', 'Extrinsics0')
                    ext_name_l = ext_name_l.replace('npy', 'txt')
                    ext_name_r = filename.replace('Disp0', 'Extrinsics1')
                    ext_name_r = ext_name_r.replace('npy', 'txt')
                    with open(ext_name_l, 'r') as f:
                        ext_l = f.readlines()
                    with open(ext_name_r, 'r') as f:
                        ext_r = f.readlines()
                    f = float(ext_l[0].split(' ')[0])
                    img_info_l['focal'] = f
                    base = abs(float(ext_l[1].split(' ')[3]) - float(ext_r[1].split(' ')[3]))
                    img_info_l['depth_factor'] = base * f

                    img_infos.append(img_info_l)
                    # img_infos.append(img_info_r)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        return img_infos
    
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt
    
    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox, tmp=False):
        """Crop from ``img``"""
        
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if tmp:
            templete = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            templete[crop_y1:crop_y2, crop_x1:crop_x2, :] = 1.0
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            return img, templete
        
        else:
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            return img
    
    def __getitem__(self, idx):

        img_file_path = self.data_infos[idx]['img_path']
        disp_path = self.data_infos[idx]['depth_map_path']
        depth_factor = self.data_infos[idx]['depth_factor']

        height=2160 
        width=3840
        image = np.fromfile(open(img_file_path, 'rb'), dtype=np.uint8).reshape(height, width, 3) / 255.0
        if self.config.get("use_rgb", False):
            image = image.astype(np.float32)[:, :, ::-1].copy()
        elif self.config.get("use_brg", False):
            image = image.astype(np.float32)[:, :, [0, 2, 1]].copy()
        elif self.config.get("use_gbr", False):
            image = image.astype(np.float32)[:, :, [1, 0, 2]].copy()
        elif self.config.get("use_rbg", False):
            image = image.astype(np.float32)[:, :, [2, 0, 1]].copy()
        elif self.config.get("use_grb", False):
            image = image.astype(np.float32)[:, :, [1, 2, 0]].copy()
        else:
            image = image.astype(np.float32)
            
        disp_gt = np.expand_dims(np.load(disp_path, mmap_mode='c'), -1)
        disp_gt = disp_gt.astype(np.float32)
        disp_gt_copy = disp_gt[:, :, 0].copy()
        
        depth_gt = depth_factor / disp_gt
        depth_gt[depth_gt > self.config.max_depth] = self.config.max_depth # for vis
        focal = self.data_infos[idx]['focal']

        
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
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'image_raw': image.copy(), 'mask_raw': mask_raw, 'image_path': img_file_path}
            
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
                x_start, y_start = [0 + 3 * self.overlap / 2, 540 + self.overlap / 2, 1080 - self.overlap / 2, 1620 - 3 * self.overlap / 2], \
                    [0 + 3 * self.overlap / 2, 960 + self.overlap / 2, 1920 - self.overlap / 2, 2880 - 3 * self.overlap / 2]
                img_crops = []
                bboxs_roi = []
                crop_areas = []
                bboxs_raw_list = []
                for x in x_start:
                    for y in y_start:
                        bbox = (int(x), int(x+540), int(y), int(y+960))
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
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': True,
                          'image_path': img_file_path, 'depth_path': disp_path, 'depth_factor_path': depth_factor,
                          'mask': mask, 'image_raw': image.copy(), 'disp_gt_edges': disp_gt_edges, 'image_path': img_file_path}
                if bboxs_roi is not None:
                    sample['bbox'] = bboxs_roi
                    sample['bbox_raw'] = bboxs_raw
                if crop_areas is not None:
                    sample['crop_area'] = crop_areas
                

            else:
                sample = {'image': image, 'focal': focal, 'image_raw': image.copy(), 'disp_gt_edges': disp_gt_edges, 'image_path': img_file_path}
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


def get_u4k_loader(config, mode, transform):
    if mode == 'train':
        dataset = U4KDataset(config, mode, config.data_path, config.filenames_train)
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
    
    elif mode == 'train_save':
        dataset = U4KDataset(config, 'online_eval', config.data_path, config.filenames_train)

        if config.distributed:
            train_sampler = None
        else:
            train_sampler = None
        
        dataloader = DataLoader(dataset, 1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=False,
                                sampler=train_sampler)
            

    elif mode == 'online_eval':
        dataset = U4KDataset(config, mode, config.data_path, config.filenames_val)
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
        dataset = U4KDataset(config, mode, config.data_path, config.filenames_test)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=1)
        
    return dataloader