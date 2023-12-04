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
from zoedepth.models.layers.swin_layers import G2LFusion
from zoedepth.models.layers.transformer import TransformerEncoder, TransformerEncoderLayer
from torchvision.ops import roi_align as torch_roi_align

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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if mid_channels is not None:
            self.conv = DoubleConvWOBN(in_channels, out_channels, mid_channels)
        else:
            self.conv = DoubleConvWOBN(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNetv1(nn.Module):
    def __init__(self, n_channels, g2l, pos_embed=False, use_area_prior=True):
        super(UNetv1, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 256)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 256)
        self.down4 = Down(256, 256)
        self.down5 = Down(256, 256)

        self.up1 = Upv1(256+256+256, 256, 384)
        self.up2 = Upv1(256+256+256, 256, 384)
        self.up3 = Upv1(256+256+256, 256, 384)
        self.up4 = Upv1(256+256+256, 256, 384)
        self.up5 = Upv1(256+32+256, 32, 272)

        self.g2l = g2l
        
        if self.g2l:
            self.g2l_att = nn.ModuleList()
            win = 12
            in_channels = [32, 256, 256, 256, 256, 256]
            crf_dims = [32, 256, 256, 256, 256, 256]

            self.g2l5 = G2LFusion(input_dim=in_channels[5], embed_dim=crf_dims[5], window_size=win, num_heads=32, depth=4, num_patches=12*16)
            self.g2l4 = G2LFusion(input_dim=in_channels[4], embed_dim=crf_dims[4], window_size=win, num_heads=32, depth=4, num_patches=24*32)
            self.g2l3 = G2LFusion(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, num_heads=16, depth=3, num_patches=48*64)
            self.g2l2 = G2LFusion(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, num_heads=16, depth=3, num_patches=96*128)
            self.g2l1 = G2LFusion(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, num_heads=8, depth=2, num_patches=192*256)
            self.g2l0 = G2LFusion(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, num_heads=8, depth=2, num_patches=384*512) 

            self.conv5 = DoubleConvWOBN(in_channels[4] * 2, in_channels[4], in_channels[4])
            self.conv4 = DoubleConvWOBN(in_channels[4] * 2, in_channels[4], in_channels[4])
            self.conv3 = DoubleConvWOBN(in_channels[3] * 2, in_channels[3], in_channels[3])
            self.conv2 = DoubleConvWOBN(in_channels[2] * 2, in_channels[2], in_channels[2])
            self.conv1 = DoubleConvWOBN(in_channels[1] * 2, in_channels[1], in_channels[1])
            self.conv0 = DoubleConvWOBN(in_channels[0] * 2, in_channels[0], in_channels[0])
            
    def forward(self, 
                input_tensor, 
                guide_plus, 
                guide_cat, 
                crop_area_resize=None, 
                bbox=None, 
                fine_feat_crop=None, 
                coarse_feat_whole=None, 
                coarse_feat_whole_hack=None, 
                coarse_feat_crop=None):

        # apply unscaled feat to swin
        if coarse_feat_whole_hack is not None:
            coarse_feat_whole = coarse_feat_whole_hack

        if crop_area_resize is None:
            not_use_prior = True
        else:
            not_use_prior = False
        
        x1 = self.inc(input_tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        if self.g2l:
            g2l_feat5 = self.g2l5(coarse_feat_whole[0], crop_area_resize[0])
            g2l_feat5 = torch_roi_align(g2l_feat5, bbox, (12, 16), 12/384, aligned=True)
            x6 = self.conv5(torch.cat([x6, g2l_feat5], dim=1))
  
        x5 = self.up1(torch.cat([x6, guide_cat[0]], dim=1), x5)
        if self.g2l:
            g2l_feat4 = self.g2l4(coarse_feat_whole[1], crop_area_resize[1])
            g2l_feat4 = torch_roi_align(g2l_feat4, bbox, (24, 32), 24/384, aligned=True)
            x5 = self.conv4(torch.cat([x5, g2l_feat4], dim=1))   

        x4 = self.up2(torch.cat([x5, guide_cat[1]], dim=1), x4)
        if self.g2l:
            g2l_feat3 = self.g2l3(coarse_feat_whole[2], crop_area_resize[2])
            g2l_feat3 = torch_roi_align(g2l_feat3, bbox, (48, 64), 48/384, aligned=True)
            x4 = self.conv3(torch.cat([x4, g2l_feat3], dim=1))

        x3 = self.up3(torch.cat([x4, guide_cat[2]], dim=1), x3)
        if self.g2l:
            g2l_feat2 = self.g2l2(coarse_feat_whole[3], crop_area_resize[3])
            g2l_feat2 = torch_roi_align(g2l_feat2, bbox, (96, 128), 96/384, aligned=True)
            x3 = self.conv2(torch.cat([x3, g2l_feat2], dim=1))

        x2 = self.up4(torch.cat([x3, guide_cat[3]], dim=1), x2)
        if self.g2l:
            g2l_feat1 = self.g2l1(coarse_feat_whole[4], crop_area_resize[4])
            g2l_feat1 = torch_roi_align(g2l_feat1, bbox, (192, 256), 192/384, aligned=True)
            x2 = self.conv1(torch.cat([x2, g2l_feat1], dim=1))

        x1 = self.up5(torch.cat([x2, guide_cat[4]], dim=1), x1)
        if self.g2l:
            g2l_feat0 = self.g2l0(coarse_feat_whole[5], crop_area_resize[5])
            g2l_feat0 = torch_roi_align(g2l_feat0, bbox, (384, 512), 384/384, aligned=True)
            x1 = self.conv0(torch.cat([x1, g2l_feat0], dim=1))

        output = [x1, x2, x3, x4, x5, x6]
        return output