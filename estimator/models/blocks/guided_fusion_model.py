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


import torch
import torch.nn as nn
import torch.nn.functional as F
# from zoedepth.models.layers.swin_layers import G2LFusion
from estimator.models.blocks.swin_layers import G2LFusion
from torchvision.ops import roi_align as torch_roi_align
from estimator.registry import MODELS

class DoubleConvWOBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Upv1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if mid_channels is not None:
            self.conv = DoubleConvWOBN(in_channels, out_channels, mid_channels)
        else:
            self.conv = DoubleConvWOBN(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

@MODELS.register_module()
class GuidedFusionPatchFusion(nn.Module):
    def __init__(
        self, 
        n_channels, 
        g2l,
        in_channels=[32, 256, 256, 256, 256, 256],
        depth=[2, 2, 3, 3, 4, 4],
        num_heads=[8, 8, 16, 16, 32, 32],
        # num_patches=[12*16, 24*32, 48*64, 96*128, 192*256, 384*512], 
        num_patches=[384*512, 192*256, 96*128, 48*64, 24*32, 12*16], 
        patch_process_shape=[384, 512]):
        
        super(GuidedFusionPatchFusion, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, in_channels[0])
        
        self.down_conv_list = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            lay = Down(in_channels[idx], in_channels[idx+1])
            self.down_conv_list.append(lay)
        
            
        in_channels_inv = in_channels[::-1]
        self.up_conv_list = nn.ModuleList()
        for idx in range(1, len(in_channels)):
            lay = Upv1(in_channels_inv[idx] + in_channels_inv[idx-1] + in_channels_inv[idx-1], in_channels_inv[idx])
            self.up_conv_list.append(lay)
            
        self.g2l = g2l
        if self.g2l:
            self.g2l_att = nn.ModuleList()
            
            win = 12
            self.patch_process_shape = patch_process_shape
            num_heads_inv = num_heads[::-1]
            depth_inv = depth[::-1]
            num_patches_inv = num_patches[::-1]
            self.g2l_list = nn.ModuleList()
            self.convs = nn.ModuleList()
            
            for idx in range(len(in_channels_inv)):
                g2l_layer = G2LFusion(input_dim=in_channels_inv[idx], embed_dim=in_channels_inv[idx], window_size=win, num_heads=num_heads_inv[idx], depth=depth_inv[idx], num_patches=num_patches_inv[idx])
                self.g2l_list.append(g2l_layer)
                layer = DoubleConvWOBN(in_channels_inv[idx] * 2, in_channels_inv[idx], in_channels_inv[idx])
                self.convs.append(layer)
                
            # self.g2l5 = G2LFusion(input_dim=in_channels[5], embed_dim=crf_dims[5], window_size=win, num_heads=32, depth=4, num_patches=num_patches[0])
            # self.g2l4 = G2LFusion(input_dim=in_channels[4], embed_dim=crf_dims[4], window_size=win, num_heads=32, depth=4, num_patches=num_patches[1])
            # self.g2l3 = G2LFusion(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, num_heads=16, depth=3, num_patches=num_patches[2])
            # self.g2l2 = G2LFusion(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, num_heads=16, depth=3, num_patches=num_patches[3])
            # self.g2l1 = G2LFusion(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, num_heads=8, depth=2, num_patches=num_patches[4])
            # self.g2l0 = G2LFusion(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, num_heads=8, depth=2, num_patches=num_patches[5]) 
            # self.conv5 = DoubleConvWOBN(in_channels[5] * 2, in_channels[5], in_channels[5])
            # self.conv4 = DoubleConvWOBN(in_channels[4] * 2, in_channels[4], in_channels[4])
            # self.conv3 = DoubleConvWOBN(in_channels[3] * 2, in_channels[3], in_channels[3])
            # self.conv2 = DoubleConvWOBN(in_channels[2] * 2, in_channels[2], in_channels[2])
            # self.conv1 = DoubleConvWOBN(in_channels[1] * 2, in_channels[1], in_channels[1])
            # self.conv0 = DoubleConvWOBN(in_channels[0] * 2, in_channels[0], in_channels[0])
            
    def forward(self, 
                input_tensor, 
                guide_plus, 
                guide_cat, 
                bbox=None, 
                fine_feat_crop=None, 
                coarse_feat_whole=None, 
                coarse_feat_whole_hack=None, 
                coarse_feat_crop=None):

        # apply unscaled feat to swin
        if coarse_feat_whole_hack is not None:
            coarse_feat_whole = coarse_feat_whole_hack

        feat_list = []
        
        x = self.inc(input_tensor)
        feat_list.append(x)
        
        for layer in self.down_conv_list:
            x = layer(x)
            feat_list.append(x)
        
        output = []
        feat_inv_list = feat_list[::-1]
        for idx, (feat_enc, feat_c) in enumerate(zip(feat_inv_list, coarse_feat_whole)):
            
            # in case for depth-anything
            _, _, h, w = feat_enc.shape
            if h != feat_c.shape[-2] or w != feat_c.shape[-1]:
                feat_enc = F.interpolate(feat_enc, size=feat_c.shape[-2:], mode='bilinear', align_corners=True)
                
            if idx == 0:
                pass
            else:
                feat_enc = self.up_conv_list[idx-1](torch.cat([temp_feat, guide_cat[idx-1]], dim=1), feat_enc)
            
            _, _, h, w = feat_c.shape
            feat_c = self.g2l_list[idx](feat_c, None)
            feat_c = torch_roi_align(feat_c, bbox, (h, w), h/self.patch_process_shape[0], aligned=True)
            x = self.convs[idx](torch.cat([feat_enc, feat_c], dim=1))
            temp_feat = x
            output.append(x)

        return output[::-1]