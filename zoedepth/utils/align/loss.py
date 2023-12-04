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

# File author: Shariq Farooq Bhat, Zhenyu Li

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    img1 = nn.functional.interpolate(img1, (256, 256), mode='bilinear', align_corners=True)
    img2 = nn.functional.interpolate(img2, (256, 256), mode='bilinear', align_corners=True)
    # h, w = 256, 256
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret



class SSIMLoss(nn.Module):
    def __init__(self, min_depth=1e-3, max_depth=10):
        super(SSIMLoss, self).__init__()
        self.name = 'SSIM'
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, input, target):
        loss =  torch.clamp((1 - ssim(input, target, val_range=self.max_depth/self.min_depth)) * 0.5, 0, 1)
        return loss


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super().__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target):
        alpha = 1e-10
        g = torch.log(input + alpha) - torch.log(target + alpha)

        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

        loss = 10 * torch.sqrt(Dg)
        return loss

def gradient_y(img):
    gy = torch.cat( [F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).to(img.device), padding=1) for i in range(img.shape[1])], 1)
    return gy

def gradient_x(img):
    gx = torch.cat( [F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).to(img.device), padding=1) for i in range(img.shape[1])], 1)
    return gx

def laplacian(img):
    lap = torch.cat( [F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view((1, 1, 3, 3)).to(img.device), padding=1) for i in range(img.shape[1])], 1)
    return lap

def laplacian_matching_loss(img1, img2, mask=None):
    return torch.mean(torch.abs(laplacian(img1)[mask] - laplacian(img2)[mask]))

class GradL1Loss(nn.Module):
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None):

        grad_gt_x = gradient_x(target)
        grad_gt_y = gradient_y(target)
        grad_pred_x = gradient_x(input)
        grad_pred_y = gradient_y(input)
        loss = torch.mean(torch.abs(grad_pred_x[mask] - grad_gt_x[mask])) + torch.mean(torch.abs(grad_pred_y[mask] - grad_gt_y[mask]))
        return loss

# Edge aware smoothness loss implementation is adapted from: https://github.com/anuragranj/cc
def edge_aware_smoothness_per_pixel(img, pred):
    """ A measure of how closely the gradients of a predicted disparity/depth map match the 
    gradients of the RGB image. 

    Args:
        img (c x 3 x h x w tensor): RGB image
        pred (c x h x w tensor): predicted depth/disparity

    Returns:
        c x 1 tensor: measure of gradient matching (smoothness loss)
    """
    

    
    pred_gradients_x = gradient_x(pred)
    pred_gradients_y = gradient_y(pred)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
            
    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)



ssim_loss = SSIMLoss()
gradl1_loss = GradL1Loss()