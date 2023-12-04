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
import torch.nn as nn
import numpy as np

from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial, ConditionalLogBinomialV2
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor, SeedBinRegressorUnnormed)
from zoedepth.models.model_io import load_state_from_resource
from torchvision.transforms import Normalize
from torchvision.ops import roi_align as torch_roi_align
from zoedepth.utils.misc import generatemask

from zoedepth.models.layers.transformer import TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder

from zoedepth.utils.misc import colorize, colors
import matplotlib.pyplot as plt

from zoedepth.models.layers.fusion_network import UNetv1
import matplotlib.pyplot as plt

import os
import torch.distributed as dist
import torch.nn.functional as F

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_activation(name, bank):
    # input of forward_hook will be a function of model/inp/oup
    def hook(module, input, output):
        bank[name] = output
    return hook

def get_input(name, bank):
    # input of forward_hook will be a function of model/inp/oup
    def hook(module, input, output):
        bank[name] = input
    return hook
    
class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

class PatchFusion(DepthModel):
    def __init__(self, coarse_model, fine_model, n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, sr_ratio=1,
                 raw_depth_shape=(2160, 3840), transform_sample_gt_size=(2160, 3840), representation='',
                 fetch_features=True, sample_feat_level=3, use_hr=False, deform=False, wo_bins=False, baseline=False,
                 condition=True, freeze=False, g2l=False, use_fusion_network=False, use_area_prior=False,
                 unet_version='v1', consistency_training=False, consistency_target='unet_feat', pos_embed=False, **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.

            sr_ratio: sr ratio during infer
            raw_depth_shape: raw depth shape during infer. times sr_ratio will be the target resolution. Used to sample points during training
            transform_sample_gt_size: training depth shape # influenced by crop shape which is not included in this pipeline right now
            representation: I use it to test the "bilap head" and a discarded idea
            fetch_features: if fetch feats. Default=True
        """
        super().__init__()

        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas
        
        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus": # default
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")
        
        N_MIDAS_OUT = 32
        btlnck_features = self.fine_model.core.output_channels[0]
        num_out_features = self.fine_model.core.output_channels[1:] # all of them are the same

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        # 1000, 2, inv, mean
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])
        
        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)
        
        self.handles = []
        self.hook_feats = {}
        self.set_fetch_features(fetch_features)

        # settings for patchfusion
        self.use_area_prior = use_area_prior
        self.g2l = g2l

        self.fusion_conv_list = nn.ModuleList()
        for i in range(6):
            if i == 5:
                layer = nn.Conv2d(N_MIDAS_OUT * 2, N_MIDAS_OUT, 3, 1, 1)
            else:
                layer = nn.Conv2d(btlnck_features * 2, btlnck_features, 3, 1, 1)
            self.fusion_conv_list.append(layer)

        self.coarse_input_proj = nn.ModuleList()
        for i in range(6):
            if i == 4:
                layer = nn.Conv2d(N_MIDAS_OUT, N_MIDAS_OUT, 3, 1, 1)
            else:
                layer = nn.Conv2d(btlnck_features, btlnck_features, 3, 1, 1)
            self.coarse_input_proj.append(layer)

        self.fine_input_proj = nn.ModuleList()
        for i in range(6):
            if i == 4:
                layer = nn.Conv2d(N_MIDAS_OUT, N_MIDAS_OUT, 3, 1, 1)
            else:
                layer = nn.Conv2d(btlnck_features, btlnck_features, 3, 1, 1)
            self.fine_input_proj.append(layer)

        self.coarse_depth_proj = nn.Conv2d(1, last_in, 3, 1, 1)
        self.fine_depth_proj = nn.Conv2d(1, last_in, 3, 1, 1)
        # self.coarse_depth_proj = nn.Conv2d(1, 1, 3, 1, 1)
        # self.fine_depth_proj = nn.Conv2d(1, 1, 3, 1, 1)
        # self.init_weight()

        self.freeze = freeze
        if self.freeze:
            # Freeze the parameters of sub_model
            for param in self.fine_model.parameters():
                param.requires_grad = False
            
            for param in self.coarse_model.parameters():
                param.requires_grad = False

            # Set the sub_model to evaluation mode
            self.fine_model.eval()
            self.coarse_model.eval()

        self.use_fusion_network = use_fusion_network
        if self.use_fusion_network:
            self.fusion_extractor = UNetv1(5, self.g2l, pos_embed, use_area_prior)

        self.consistency_training = consistency_training
        if self.consistency_training:
            if consistency_target == 'mix':
                consistency_target = 'unet_feat'
            self.consistency_target = consistency_target
            print("current consistency target is {}".format(consistency_target))

            self.consistency_projs = nn.ModuleList()
            if self.consistency_target == 'unet_feat':
                for i in range(6):
                    if i == 5:
                        layer = nn.Conv2d(N_MIDAS_OUT, N_MIDAS_OUT, 1, 1, 0)
                        layer = nn.Identity()
                    else:
                        layer = nn.Conv2d(btlnck_features, btlnck_features, 1, 1, 0)
                        layer = nn.Identity()
                    self.consistency_projs.append(layer)
            
            if self.consistency_target == 'final_feat':
                layer = nn.Conv2d(64, 64, 1, 1, 0) # 192
                layer = nn.Identity()
                self.consistency_projs.append(layer)
                layer = nn.Conv2d(32, 32, 1, 1, 0) # 384
                layer = nn.Identity()
                self.consistency_projs.append(layer)
                layer = nn.Conv2d(128, 128, 1, 1, 0) # 192
                layer = nn.Identity()
                self.consistency_projs.append(layer)
                
    def init_weight(self):
        for m in self.coarse_input_proj:
            torch.nn.init.constant_(m.weight, 0)
            torch.nn.init.constant_(m.bias, 0)

        for m in self.fine_input_proj:
            torch.nn.init.constant_(m.weight, 0)
            torch.nn.init.constant_(m.bias, 0)

        torch.nn.init.constant_(self.coarse_depth_proj.weight, 0)
        torch.nn.init.constant_(self.fine_depth_proj.weight, 0)

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        
        # return self.fusion_extractor.parameters()
    
        param_conf = []
        param_conf_coarse_model = self.coarse_model.get_lr_params(lr)
        param_conf_fine_model = self.fine_model.get_lr_params(lr)
        param_conf.extend(param_conf_coarse_model)
        param_conf.extend(param_conf_fine_model)


        skip_list = {'absolute_pos_embed'}
        skip_keywords = {'relative_position_bias_table'}
        skip_hack = {'g2l0', 'g2l1', 'g2l2', 'g2l3', 'g2l4', 'g2l5'}

        no_decay = []
        has_decay = []
        fusion_enc = []

        for name, param in self.named_parameters():
            if 'coarse_model' not in name and 'fine_model' not in name:
                if len(param.shape) == 1 or (name.endswith(".bias") and check_keywords_in_name(name, skip_hack)) or check_keywords_in_name(name, skip_list) or \
                        check_keywords_in_name(name, skip_keywords):
                    # print("no decay: {}".format(name))
                    no_decay.append(param)
                else:
                    # print("has decay: {}".format(name))
                    has_decay.append(param)
        
        param_conf.append({'params': has_decay, 'lr': lr})
        param_conf.append({'params': no_decay, 'weight_decay': 0., 'lr': lr})
        # param_conf.append({'params': fusion_enc, 'lr': lr / self.encoder_lr_factor})
        param_conf.append({'params': fusion_enc, 'lr': lr})

        return param_conf
    
    def forward(
        self, 
        x, 
        sampled_depth=None, 
        mode='train', 
        return_final_centers=False, 
        denorm=False, 
        return_probs=False, 
        image_raw=None, 
        bbox=None, 
        crop_area=None, 
        shift=None, 
        bbox_raw=None, 
        iter_prior=None,
        previous_info=None,
        **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.
        
        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """

        if self.consistency_training and mode == 'train':
            split_x = torch.split(x, 3, dim=1)
            x = torch.cat(split_x, dim=0)
            image_raw = torch.cat([image_raw, image_raw], dim=0)
            split_bbox = torch.split(bbox, 4, dim=-1)
            bbox = torch.cat(split_bbox, dim=0)
            split_bbox = torch.split(bbox_raw, 4, dim=-1)
            bbox_raw = torch.cat(split_bbox, dim=0)
            crop_area = torch.split(crop_area, 1, dim=1)
            crop_area = torch.cat(crop_area, dim=0)

        crop_input = x
    
        # coarse forward
        if self.freeze:
            with torch.no_grad():
                if self.fine_model.training:
                    self.fine_model.eval()
                    self.coarse_model.eval()
                    
                if previous_info is None:
                    previous_info = dict()
                    whole_depth_pred = self.coarse_model(image_raw)['metric_depth']
                    previous_info['whole_depth_pred'] = whole_depth_pred
                    previous_info['coarse_model.hook_feats'] = self.coarse_model.hook_feats
                    
                else:
                    whole_depth_pred = previous_info['whole_depth_pred']
                    self.coarse_model.hook_feats = dict()
                    self.coarse_model.hook_feats = previous_info['coarse_model.hook_feats']
                
                fine_depth_pred = self.fine_model(x)['metric_depth']

                whole_depth_pred = nn.functional.interpolate(
                    whole_depth_pred, (2160, 3840), mode='bilinear', align_corners=True)

        else:
            whole_depth_pred = self.coarse_model(image_raw)['metric_depth']
            fine_depth_pred = self.fine_model(x)['metric_depth']
        
        coarse_model_midas_enc_feats = [
            self.coarse_input_proj[5](self.coarse_model.hook_feats['x_d0']),
            self.coarse_input_proj[0](self.coarse_model.hook_feats['x_blocks_feat_0']), 
            self.coarse_input_proj[1](self.coarse_model.hook_feats['x_blocks_feat_1']), 
            self.coarse_input_proj[2](self.coarse_model.hook_feats['x_blocks_feat_2']), 
            self.coarse_input_proj[3](self.coarse_model.hook_feats['x_blocks_feat_3']),
            self.coarse_input_proj[4](self.coarse_model.hook_feats['midas_final_feat'])] # 384

        if self.g2l:
            coarse_model_midas_enc_feats_g2l = coarse_model_midas_enc_feats
                
        if self.use_area_prior:
            crop_area_resize = [
                nn.functional.interpolate(crop_area, (12, 16), mode='bilinear', align_corners=True),
                nn.functional.interpolate(crop_area, (24, 32), mode='bilinear', align_corners=True),
                nn.functional.interpolate(crop_area, (48, 64), mode='bilinear', align_corners=True),
                nn.functional.interpolate(crop_area, (96, 128), mode='bilinear', align_corners=True),
                nn.functional.interpolate(crop_area, (192, 256), mode='bilinear', align_corners=True),
                nn.functional.interpolate(crop_area, (384, 512), mode='bilinear', align_corners=True)]
        else:
            crop_area_resize = None

        inds = torch.arange(bbox.shape[0]).to(bbox.device).unsqueeze(dim=-1)
        bbox = torch.cat((inds, bbox), dim=-1)
        coarse_model_midas_enc_roi_feats = [
            torch_roi_align(coarse_model_midas_enc_feats[0], bbox, (12, 16), 12/384, aligned=True),
            torch_roi_align(coarse_model_midas_enc_feats[1], bbox, (24, 32), 24/384, aligned=True),
            torch_roi_align(coarse_model_midas_enc_feats[2], bbox, (48, 64), 48/384, aligned=True),
            torch_roi_align(coarse_model_midas_enc_feats[3], bbox, (96, 128), 96/384, aligned=True),
            torch_roi_align(coarse_model_midas_enc_feats[4], bbox, (192, 256), 192/384, aligned=True),
            torch_roi_align(coarse_model_midas_enc_feats[5], bbox, (384, 512), 384/384, aligned=True)
        ]

        # whole_depth_roi_pred = torch_roi_align(whole_depth_pred, bbox, (384, 512), 384/384)
        # back to full resolution to avoid potential misalignment
        bbox_hack = copy.deepcopy(bbox)
        bbox_hack[:, 1] = bbox[:, 1] / 512 * 3840 # scale back to full resolution coord
        bbox_hack[:, 2] = bbox[:, 2] / 384 * 2160
        bbox_hack[:, 3] = bbox[:, 3] / 512 * 3840
        bbox_hack[:, 4] = bbox[:, 4] / 384 * 2160
        whole_depth_roi_pred = torch_roi_align(whole_depth_pred, bbox_hack, (384, 512), 1, aligned=True)

        fine_model_midas_enc_feats = [
            self.fine_input_proj[5](self.fine_model.hook_feats['x_d0']),
            self.fine_input_proj[0](self.fine_model.hook_feats['x_blocks_feat_0']), 
            self.fine_input_proj[1](self.fine_model.hook_feats['x_blocks_feat_1']), 
            self.fine_input_proj[2](self.fine_model.hook_feats['x_blocks_feat_2']), 
            self.fine_input_proj[3](self.fine_model.hook_feats['x_blocks_feat_3']),
            self.fine_input_proj[4](self.fine_model.hook_feats['midas_final_feat'])] # 384
        
        x_plane = []
        x_blocks = []
        feat_plus_list = []
        feat_cat_list = []
        res_pool = [(24, 32), (48, 64), (96, 128), (192, 256), (384, 512)]
        for l_i, (f_ca, f_c_roi, f_f) in enumerate(zip(coarse_model_midas_enc_feats, coarse_model_midas_enc_roi_feats, fine_model_midas_enc_feats)):
            feat_cat = self.fusion_conv_list[l_i](torch.cat([f_c_roi, f_f], dim=1))
            feat_plus = f_c_roi + f_f
            feat_cat_list.append(feat_cat)
            feat_plus_list.append(feat_plus)

        if iter_prior is not None:
            input_tensor = torch.cat([whole_depth_roi_pred, iter_prior, crop_input], dim=1)
        else:
            input_tensor = torch.cat([whole_depth_roi_pred, fine_depth_pred, crop_input], dim=1)
        output = self.fusion_extractor(
            input_tensor = input_tensor,
            guide_plus = feat_plus_list,
            guide_cat = feat_cat_list,
            bbox = bbox,
            crop_area_resize = crop_area_resize,
            fine_feat_crop = fine_model_midas_enc_feats,
            coarse_feat_whole = coarse_model_midas_enc_feats,
            coarse_feat_crop = coarse_model_midas_enc_roi_feats,
            coarse_feat_whole_hack=None)[::-1] # low -> high
 
        x_blocks = output
        x = x_blocks[0]
        x_blocks = x_blocks[1:]

        if self.consistency_training:
            if self.consistency_target == 'unet_feat':
                proj_feat_list = []
                for idx, feat in enumerate(output):
                    proj_feat = self.consistency_projs[idx](feat)
                    proj_feat_list.append(proj_feat)

        # NOTE: below is ZoeDepth implementation
        # # new last
        # last = coarse_model_midas_enc_roi_feats[-1] + fine_model_midas_enc_feats[-1]
        last = x_blocks[-1] # have already been fused in x_blocks
        self.hook_feats['midas_final_feat'] = last

        bs, c, h, w = last.shape
        rel_cond = torch.zeros((bs, 1, h, w), device=last.device)

        self.hook_feats['rel_depth'] = rel_cond # skip this

        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for idx, (projector, attractor, x) in enumerate(zip(self.projectors, self.attractors, x_blocks)):
            b_embedding = projector(x)
            self.hook_feats['x_blocks_feat_{}'.format(idx)] = x
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        self.hook_feats['b_centers'] = b_centers

        
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

            
        final_pred = out
        output = dict(metric_depth=final_pred)

        output['coarse_depth_pred'] = whole_depth_pred
        output['fine_depth_pred'] = fine_depth_pred
        output['coarse_depth_pred_roi'] = whole_depth_roi_pred
        if self.consistency_training:
            if self.consistency_target == 'final_feat' or self.consistency_target == 'unet_feat':
                output['temp_features'] = proj_feat_list
        
        output['previous_info'] = previous_info
            
        return output

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, coarse_model_path=None, fine_model_path=None, **kwargs):
        from zoedepth.models.zoedepth_custom.zoedepth_custom import ZoeDepthCustom
        
        print("build pretrained condition model from {}".format(coarse_model_path))
        coarse_model = ZoeDepthCustom.build(
            midas_model_type=midas_model_type, 
            pretrained_resource=coarse_model_path, 
            # pretrained_resource="", 
            use_pretrained_midas=use_pretrained_midas, 
            # use_pretrained_midas=False, 
            train_midas=train_midas, 
            freeze_midas_bn=freeze_midas_bn, 
            **kwargs)

        print("build pretrained condition model from {}".format(fine_model_path))
        fine_model = ZoeDepthCustom.build(
            midas_model_type=midas_model_type, 
            pretrained_resource=fine_model_path, 
            # pretrained_resource="", 
            use_pretrained_midas=use_pretrained_midas,
            # use_pretrained_midas=False,
            train_midas=train_midas, 
            freeze_midas_bn=freeze_midas_bn, 
            **kwargs)
        
        model = PatchFusion(coarse_model, fine_model, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)

        return model
    
    @staticmethod
    def build_from_config(config):
        return PatchFusion.build(**config)
 
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self
    
    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks()
        else:
            self.remove_hooks()
        return self
    
    def attach_hooks(self):
        
        self.handles.append(self.seed_projector.register_forward_hook(get_activation("seed_projector", self.hook_feats)))
        
        for idx, proj in enumerate(self.projectors):
            self.handles.append(proj.register_forward_hook(get_activation("projector_{}".format(idx), self.hook_feats)))
            
        for idx, proj in enumerate(self.attractors):
            self.handles.append(proj.register_forward_hook(get_activation("attractor_{}".format(idx), self.hook_feats)))

        return self
