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

import itertools

import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmengine import print_log
from mmengine.config import ConfigDict
from torchvision.ops import roi_align as torch_roi_align
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig

from estimator.registry import MODELS
from estimator.models import build_model
from estimator.models.baseline_pretrain import BaselinePretrain
from estimator.models.utils import generatemask

from zoedepth.models.zoedepth import ZoeDepth
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor, SeedBinRegressorUnnormed)
from zoedepth.models.base_models.midas import Resize as ResizeZoe
from depth_anything.transform import Resize as ResizeDA



@MODELS.register_module()
class PatchFusion(BaselinePretrain, PyTorchModelHubMixin):
    def __init__(
        self, 
        config,):
        """ZoeDepth model
        """
        nn.Module.__init__(self)
        
        if isinstance(config, ConfigDict):
            # convert a ConfigDict to a PretrainedConfig for hf saving
            config = PretrainedConfig.from_dict(config.to_dict())
            config.load_branch = True
        else:
            # used when loading patchfusion from hf model space
            config = PretrainedConfig.from_dict(ConfigDict(**config).to_dict())
            config.load_branch = False
            config.coarse_branch.pretrained_resource = None
            config.fine_branch.pretrained_resource = None
            
        self.config = config
        
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        
        self.patch_process_shape = config.patch_process_shape
        self.tile_cfg = self.prepare_tile_cfg(config.image_raw_shape, config.patch_split_num)
        
        self.coarse_branch_cfg = config.coarse_branch
        if config.coarse_branch.type == 'ZoeDepth':
            self.coarse_branch = ZoeDepth.build(**config.coarse_branch)
            self.resizer = ResizeZoe(config.patch_process_shape[1], config.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
        elif config.coarse_branch.type == 'DA-ZoeDepth':
            self.coarse_branch = ZoeDepth.build(**config.coarse_branch)
            self.resizer = ResizeDA(config.patch_process_shape[1], config.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
        else:
            raise NotImplementedError
        
        if config.fine_branch.type == 'ZoeDepth':
            self.fine_branch = ZoeDepth.build(**config.fine_branch)
        elif config.fine_branch.type == 'DA-ZoeDepth':
            self.fine_branch = ZoeDepth.build(**config.fine_branch)
        else:
            raise NotImplementedError
        
        if config.load_branch:
            print_log("Loading coarse_branch from {}".format(config.pretrain_model[0]), logger='current') 
            print_log(self.coarse_branch.load_state_dict(torch.load(config.pretrain_model[0], map_location='cpu')['model_state_dict'], strict=True), logger='current') # coarse ckp
            print_log("Loading fine_branch from {}".format(config.pretrain_model[1]), logger='current')
            print_log(self.fine_branch.load_state_dict(torch.load(config.pretrain_model[1], map_location='cpu')['model_state_dict'], strict=True), logger='current')
        
        # freeze all these parameters
        for param in self.coarse_branch.parameters():
            param.requires_grad = False
        for param in self.fine_branch.parameters():
            param.requires_grad = False
                
        self.sigloss = build_model(config.sigloss)
        
        N_MIDAS_OUT = 32
        btlnck_features = self.fine_branch.core.output_channels[0]
        self.fusion_conv_list = nn.ModuleList()
        for i in range(6):
            if i == 5:
                layer = nn.Conv2d(N_MIDAS_OUT * 2, N_MIDAS_OUT, 3, 1, 1)
            else:
                layer = nn.Conv2d(btlnck_features * 2, btlnck_features, 3, 1, 1)
            self.fusion_conv_list.append(layer)

        self.guided_fusion = build_model(config.guided_fusion)
        
        # NOTE: a decoder head
        if self.coarse_branch_cfg.bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif self.coarse_branch_cfg.bin_centers_type == "softplus": # default
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif self.coarse_branch_cfg.bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif self.coarse_branch_cfg.bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")
        
        N_MIDAS_OUT = 32
        btlnck_features = self.fine_branch.core.output_channels[0]
        num_out_features = self.fine_branch.core.output_channels[1:] # all of them are the same

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=self.coarse_branch_cfg.n_bins, min_depth=config.min_depth, max_depth=config.max_depth)
        self.seed_projector = Projector(btlnck_features, self.coarse_branch_cfg.bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, self.coarse_branch_cfg.bin_embedding_dim)
            for num_out in num_out_features
        ])
        # 1000, 2, inv, mean
        self.attractors = nn.ModuleList([
            Attractor(self.coarse_branch_cfg.bin_embedding_dim, self.coarse_branch_cfg.n_bins, n_attractors=self.coarse_branch_cfg.n_attractors[i], min_depth=config.min_depth, max_depth=config.max_depth,
                      alpha=self.coarse_branch_cfg.attractor_alpha, gamma=self.coarse_branch_cfg.attractor_gamma, kind=self.coarse_branch_cfg.attractor_kind, attractor_type=self.coarse_branch_cfg.attractor_type)
            for i in range(len(num_out_features))
        ])
        
        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, self.coarse_branch_cfg.bin_embedding_dim, n_classes=self.coarse_branch_cfg.n_bins, min_temp=self.coarse_branch_cfg.min_temp, max_temp=self.coarse_branch_cfg.max_temp)
        
        # NOTE: consistency training
        self.consistency_training = False
    
      
    def load_dict(self, dict):
        return self.load_state_dict(dict, strict=False)
                
    def get_save_dict(self):
        current_model_dict = self.state_dict()
        save_state_dict = {}
        for k, v in current_model_dict.items():
            if 'coarse_branch' in k or 'fine_branch' in k:
                pass
            else:
                save_state_dict[k] = v
        return save_state_dict
    
    def coarse_forward(self, image_lr):
        with torch.no_grad():
            if self.coarse_branch.training:
                self.coarse_branch.eval()
                    
            deep_model_output_dict = self.coarse_branch(image_lr, return_final_centers=True)
            deep_features = deep_model_output_dict['temp_features'] # x_d0 1/128, x_blocks_feat_0 1/64, x_blocks_feat_1 1/32, x_blocks_feat_2 1/16, x_blocks_feat_3 1/8, midas_final_feat 1/4 [based on 384x4, 512x4]
            coarse_prediction = deep_model_output_dict['metric_depth']
            
            coarse_features = [
                deep_features['x_d0'],
                deep_features['x_blocks_feat_0'],
                deep_features['x_blocks_feat_1'],
                deep_features['x_blocks_feat_2'],
                deep_features['x_blocks_feat_3'],
                deep_features['midas_final_feat']] # bs, c, h, w

            return coarse_prediction, coarse_features
    
    def fine_forward(self, image_hr_crop):
        with torch.no_grad():
            if self.fine_branch.training:
                self.fine_branch.eval()
            
            deep_model_output_dict = self.fine_branch(image_hr_crop, return_final_centers=True)
            deep_features = deep_model_output_dict['temp_features'] # x_d0 1/128, x_blocks_feat_0 1/64, x_blocks_feat_1 1/32, x_blocks_feat_2 1/16, x_blocks_feat_3 1/8, midas_final_feat 1/4 [based on 384x4, 512x4]
            fine_prediction = deep_model_output_dict['metric_depth']
            
            fine_features = [
                deep_features['x_d0'],
                deep_features['x_blocks_feat_0'],
                deep_features['x_blocks_feat_1'],
                deep_features['x_blocks_feat_2'],
                deep_features['x_blocks_feat_3'],
                deep_features['midas_final_feat']] # bs, c, h, w
            
            return fine_prediction, fine_features
    
    def coarse_postprocess_train(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):

        coarse_features_patch_area = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            cur_lvl_feat = torch_roi_align(feat, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_features_patch_area.append(cur_lvl_feat)

        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)

        return coarse_prediction_roi, coarse_features_patch_area
    

    def coarse_postprocess_test(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):
        patch_num = bboxs_feat.shape[0]

        coarse_features_patch_area = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            feat_extend = feat.repeat(patch_num, 1, 1, 1)
            cur_lvl_feat = torch_roi_align(feat_extend, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_features_patch_area.append(cur_lvl_feat)
        
        coarse_prediction = coarse_prediction.repeat(patch_num, 1, 1, 1)
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)

        return_dict = {
            'coarse_depth_roi': coarse_prediction_roi,
            'coarse_feats_roi': coarse_features_patch_area}
        
        return return_dict
    
    def fusion_forward(self, fine_depth_pred, crop_input, coarse_model_midas_enc_feats, fine_model_midas_enc_feats, bbox_feat, coarse_depth_roi=None, coarse_feats_roi=None):
        feat_cat_list = []
        feat_plus_list = []
        
        for l_i, (f_ca, f_c_roi, f_f) in enumerate(zip(coarse_model_midas_enc_feats, coarse_feats_roi, fine_model_midas_enc_feats)):
            feat_cat = self.fusion_conv_list[l_i](torch.cat([f_c_roi, f_f], dim=1))
            feat_plus = f_c_roi + f_f
            feat_cat_list.append(feat_cat)
            feat_plus_list.append(feat_plus)
        
        input_tensor = torch.cat([coarse_depth_roi, fine_depth_pred, crop_input], dim=1)
        
        # HACK: hack for depth-anything
        # if self.coarse_branch_cfg.type == 'DA-ZoeDepth':
        #     input_tensor = F.interpolate(input_tensor, size=(448, 592), mode='bilinear', align_corners=True)
            
        output = self.guided_fusion(
            input_tensor = input_tensor,
            guide_plus = feat_plus_list,
            guide_cat = feat_cat_list,
            bbox = bbox_feat,
            fine_feat_crop = fine_model_midas_enc_feats,
            coarse_feat_whole = coarse_model_midas_enc_feats,
            coarse_feat_crop = coarse_feats_roi,
            coarse_feat_whole_hack=None)[::-1] # low -> high
            
        x_blocks = output
        x = x_blocks[0]
        x_blocks = x_blocks[1:]

        proj_feat_list = []
        if self.consistency_training:
            if self.consistency_target == 'unet_feat':
                proj_feat_list = []
                for idx, feat in enumerate(output):
                    proj_feat = self.consistency_projs[idx](feat)
                    proj_feat_list.append(proj_feat)

        # NOTE: below is ZoeDepth implementation
        last = x_blocks[-1] # have already been fused in x_blocks
        bs, c, h, w = last.shape
        rel_cond = torch.zeros((bs, 1, h, w), device=last.device)
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.coarse_branch_cfg.bin_centers_type == 'normed' or self.coarse_branch_cfg.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for idx, (projector, attractor, x) in enumerate(zip(self.projectors, self.attractors, x_blocks)):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        if self.consistency_training:
            if self.consistency_target == 'final_feat':
                proj_feat_1 = self.consistency_projs[0](b_centers)
                proj_feat_2 = self.consistency_projs[1](last)
                proj_feat_3 = self.consistency_projs[2](b_embedding)
                proj_feat_list = [proj_feat_1, proj_feat_2, proj_feat_3]

        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1) # + self.coarse_depth_proj(whole_depth_roi_pred) + self.fine_depth_proj(fine_depth_pred)
        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        # till here, we have features (attached with a relative depth prediction) and embeddings
        # post process
        # final_pred = out * self.blur_mask + whole_depth_roi_pred * (1-self.blur_mask)
        # out = F.interpolate(out, (540, 960), mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)

        out = torch.sum(x * b_centers, dim=1, keepdim=True)
        return out, proj_feat_list
    
    
    def infer_forward(self, imgs_crop, bbox_feat_forward, tile_temp, coarse_temp_dict):
        
        fine_prediction, fine_features = self.fine_forward(imgs_crop)
        
        depth_prediction, consistency_target = \
            self.fusion_forward(
                fine_prediction, 
                imgs_crop, 
                tile_temp['coarse_features'], 
                fine_features, 
                bbox_feat_forward,
                **coarse_temp_dict)
            
        return depth_prediction
    
    
    def forward(
        self,
        mode,
        image_lr,
        image_hr,
        depth_gt=None,
        crops_image_hr=None,
        crop_depths=None,
        bboxs=None,
        tile_cfg=None,
        cai_mode='m1',
        process_num=4):
        
        if mode == 'train':
            bboxs_feat_factor = torch.tensor([
                1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0], 
                1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)
            bboxs_feat = bboxs * bboxs_feat_factor
            inds = torch.arange(bboxs.shape[0]).to(bboxs.device).unsqueeze(dim=-1)
            bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
        
            coarse_prediction, coarse_features = self.coarse_forward(image_lr)
            fine_prediction, fine_features = self.fine_forward(crops_image_hr)
            coarse_prediction_roi, coarse_features_patch_area = self.coarse_postprocess_train(coarse_prediction, coarse_features, bboxs, bboxs_feat)

            depth_prediction, consistency_target = self.fusion_forward(
                fine_prediction, 
                crops_image_hr, 
                coarse_features, 
                fine_features, 
                bboxs_feat,
                coarse_depth_roi=coarse_prediction_roi,
                coarse_feats_roi=coarse_features_patch_area,)

            loss_dict = {}
            loss_dict['sig_loss'] = self.sigloss(depth_prediction, crop_depths, self.min_depth, self.max_depth)
            loss_dict['total_loss'] = loss_dict['sig_loss']
            
            return loss_dict, {'rgb': crops_image_hr, 'depth_pred': depth_prediction, 'depth_gt': crop_depths}
        
        else:
            if tile_cfg is None:
                tile_cfg = self.tile_cfg
            else:
                tile_cfg = self.prepare_tile_cfg(tile_cfg['image_raw_shape'], tile_cfg['patch_split_num'])
            
            assert image_hr.shape[0] == 1
            
            coarse_prediction, coarse_features = self.coarse_forward(image_lr)
            
            tile_temp = {
                'coarse_prediction': coarse_prediction,
                'coarse_features': coarse_features,}
            
            blur_mask = generatemask((self.patch_process_shape[0], self.patch_process_shape[1])) + 1e-3
            blur_mask = torch.tensor(blur_mask, device=image_hr.device)
            avg_depth_map = self.regular_tile(
                offset=[0, 0], 
                offset_process=[0, 0], 
                image_hr=image_hr[0], 
                init_flag=True, 
                tile_temp=tile_temp, 
                blur_mask=blur_mask,
                tile_cfg=tile_cfg,
                process_num=process_num)

            if cai_mode == 'm2' or cai_mode[0] == 'r':
                avg_depth_map = self.regular_tile(
                    offset=[0, tile_cfg['patch_raw_shape'][1]//2], 
                    offset_process=[0, self.patch_process_shape[1]//2], 
                    image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                avg_depth_map = self.regular_tile(
                    offset=[tile_cfg['patch_raw_shape'][0]//2, 0],
                    offset_process=[self.patch_process_shape[0]//2, 0], 
                    image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                avg_depth_map = self.regular_tile(
                    offset=[tile_cfg['patch_raw_shape'][0]//2, tile_cfg['patch_raw_shape'][1]//2],
                    offset_process=[self.patch_process_shape[0]//2, self.patch_process_shape[1]//2], 
                    init_flag=False, image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                
            if cai_mode[0] == 'r':
                blur_mask = generatemask((tile_cfg['patch_raw_shape'][0], tile_cfg['patch_raw_shape'][1])) + 1e-3
                blur_mask = torch.tensor(blur_mask, device=image_hr.device)
                avg_depth_map.resize(tile_cfg['image_raw_shape'])
                patch_num = int(cai_mode[1:]) // process_num
                for i in range(patch_num):
                    avg_depth_map = self.random_tile(
                        image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)

            depth = avg_depth_map.average_map
            depth = depth.unsqueeze(dim=0).unsqueeze(dim=0)

            return depth, {'rgb': image_lr, 'depth_pred': depth, 'depth_gt': depth_gt}

