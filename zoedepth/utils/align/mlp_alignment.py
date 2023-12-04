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
from tqdm.auto import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler
from zoedepth.utils.align.loss import SILogLoss, gradl1_loss, edge_aware_smoothness_per_pixel
# from utils.misc import *
from .depth_alignment import apply_depth_smoothing, scale_shift_linear
import cv2
import numpy as np

def as_bchw_tensor(input_tensor, num, device=None):
    if len(input_tensor.shape) == 2:
        input_tensor = torch.tensor(input_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
    elif len(input_tensor.shape) == 3:
        input_tensor = torch.tensor(input_tensor).unsqueeze(dim=0)
    else:
        input_tensor = input_tensor
    if device is not None:
        input_tensor = input_tensor.to(device)
    return input_tensor


def get_mlp(in_channels, out_channels):
    conv_config = dict(kernel_size=1, padding=0, stride=1)
    net =  nn.Sequential(
        # nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, stride=1),
        # nn.GELU(),
        nn.Conv2d(in_channels, 64, **conv_config),
        nn.GELU(),
        nn.Conv2d(64, 128, **conv_config),
        nn.GELU(),
        nn.Conv2d(128, out_channels, **conv_config),
    )

    # initialize last layer to predict zeroes
    # net[-1].weight.data.zero_()
    # net[-1].bias.data.zero_()
    return net

def smoothness_loss(depth):
    depth_dx = depth[:, :, :-1, :-1] - depth[:, :, :-1, 1:]
    depth_dy = depth[:, :, :-1, :-1] - depth[:, :, 1:, :-1]
    depth_dx = depth_dx.abs().mean()
    depth_dy = depth_dy.abs().mean()
    return depth_dx + depth_dy

def curvature_loss(depth):
    depth_dx = depth[:, :, :-1, :-1] - depth[:, :, :-1, 1:]
    depth_dy = depth[:, :, :-1, :-1] - depth[:, :, 1:, :-1]
    depth_dxx = depth_dx[:, :, :, :-1] - depth_dx[:, :, :, 1:]
    depth_dyy = depth_dy[:, :, :-1, :] - depth_dy[:, :, 1:, :]
    depth_dxy = depth_dx[:, :, :-1, :-1] - depth_dx[:, :, 1:, 1:]
    depth_dxx = depth_dxx.abs().mean()
    depth_dyy = depth_dyy.abs().mean()
    depth_dxy = depth_dxy.abs().mean()
    return depth_dxx + depth_dyy + depth_dxy

def multi_scale_curvature_loss(depth, scales=[1, 2, 4]):
    loss = 0
    for s in scales:
        loss += curvature_loss(F.interpolate(depth, scale_factor=1/s, mode='bilinear', align_corners=False))
    return loss


def tv_loss(x):
    """Total variation loss."""
    b, c, h, w = x.shape
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return torch.sum(dh) + torch.sum(dw)

def scale_invariant_gradient_loss(pred, gt):
    alpha = 1e-10
    kernel_grad_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).to(pred.device)
    kernel_grad_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).to(pred.device)

    g = torch.log(pred + alpha) - torch.log(gt + alpha)
    g_x = F.conv2d(g, kernel_grad_x, padding=1)
    g_y = F.conv2d(g, kernel_grad_y, padding=1)
    # n, c, h, w = g.shape
    # norm = 1/(h*w)
    # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

    Dgx = torch.var(g_x) + 0.5 * torch.pow(torch.mean(g_x), 2)
    Dgy = torch.var(g_y) + 0.5 * torch.pow(torch.mean(g_y), 2)


    loss = 10 * torch.sqrt(Dgx) + 10 * torch.sqrt(Dgy)
    return loss


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(np.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def gaussian_rff(d_model, height, width, sigma=10):
    assert d_model % 2 == 0
    B = torch.randn((d_model//2, 2)) * sigma
    
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)

    x, y = torch.meshgrid(x, y)
    xy = torch.stack([x, y], dim=-1).view(-1, 2)

    xy = torch.matmul(B, xy.T).T
    xy = 2 * np.pi * xy.view(height, width, d_model//2)
    enc = torch.cat([torch.sin(xy), torch.cos(xy)], dim=-1)
    return enc.permute(2, 0, 1)


def get_depth(pred, D):
    a, b, c = torch.split(pred, 1, dim=1)
    return 1e-7 + torch.relu(torch.exp(a) * D + (torch.sigmoid(c)-0.5)*torch.exp(b))
    # return 1e-4 + torch.exp(a) * D + b
    # return nn.Softplus()(a)


def train_mlp(image, mask, dr, dp, lr=3e-2, num_iters=3000, device='cuda:0', pos_dim=32, loss_config=dict(beta=0.99),
               w_smooth=1, w_curvature=0.0, w_gl1=0.1, w_tv=0.1, w_shift_reg=0.1, **kwargs):

    mlp = get_mlp(pos_dim+4, 3)
    # mlp = get_mlp(4, 3)
    mlp = mlp.to(device)

    optimizer = optim.AdamW(mlp.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_iters, steps_per_epoch=1)

    image = as_bchw_tensor(image, 3, device=device).detach()
    h, w = image.shape[-2:]
    pe = positionalencoding2d(pos_dim, h, w)
    pe = as_bchw_tensor(pe, pos_dim, device=device)
    D = as_bchw_tensor(dp, 1, device=device).detach()
    # pe = as_bchw_tensor(gaussian_rff(pos_dim, h, w, sigma=5), pos_dim, device=device)
    X = torch.cat([image, D, pe], dim=1)  # bchw
    # X = torch.cat([image, D], dim=1)  # bchw
    
    Y = as_bchw_tensor(dr, 1, device=device).detach()
    mask = as_bchw_tensor(mask, 1, device=device).detach()
    pbar = tqdm(range(num_iters), desc=f"Training")
    # beta_min, beta_max = 0.
    si_log = SILogLoss(**loss_config)

    for i in pbar:
        optimizer.zero_grad()
        # pred = dr.max().item() * torch.sigmoid(mlp(X))
        pred = mlp(X)
        a, b, c = torch.split(pred, 1, dim=1)
        pred = get_depth(pred, D.detach())
        loss_si = si_log(pred[mask], Y[mask])
        loss = loss_si + w_curvature * multi_scale_curvature_loss(pred) + w_gl1 * gradl1_loss(pred, D.detach()) + w_smooth * edge_aware_smoothness_per_pixel(image, pred)
        # loss_tv = w_tv * (tv_loss(a) + tv_loss(b) + tv_loss(c))
        # loss_gl1 = w_gl1 * gradl1_loss(pred, D.detach())
        # loss_gl1 = w_gl1 * scale_invariant_gradient_loss(pred, D.detach())
        # loss_shift_reg = w_shift_reg * torch.mean(b**2)
        # loss = loss_si + loss_gl1
        # loss = F.mse_loss(pred[mask], Y[mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item(), si=loss_si.item())

    return mlp

def predict_aligned(mlp, image, dp, pos_dim=32, **kwargs):
    device = next(mlp.parameters()).device
    image = as_bchw_tensor(image, 3, device=device)
    h, w = image.shape[-2:]
    pe = positionalencoding2d(pos_dim, h, w)
    pe = as_bchw_tensor(pe, pos_dim, device=device)
    D = as_bchw_tensor(dp, 1, device=device)
    # pe = as_bchw_tensor(gaussian_rff(pos_dim, h, w, sigma=5), pos_dim, device=device)

    X = torch.cat([image, D, pe], dim=1)  # bchw
    # X = torch.cat([image, D], dim=1)  # bchw
    pred = mlp(X)
    pred = get_depth(pred, D)
    return pred.detach()

def align_by_mlp(image, mask, dr, dp, **kwargs):
    mlp = train_mlp(image, mask, dr, dp, **kwargs)
    pred = predict_aligned(mlp, image, dp, **kwargs)
    return pred

from abc import ABC, abstractmethod


# Abstract class for depth alignment. All depth alignment methods should inherit from this class.
# The abstract class defines the interface for depth alignment.
class DepthAligner(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def align(self, depth_src, depth_target, valid_mask, *args, **kwargs):
        """
        Aligns the depth_src to the depth_target such that the aligned depth_src is as close as possible to the depth_target.
        """
        raise NotImplementedError
    
class MLPAligner(DepthAligner):
    def __init__(self):
        super().__init__()

    def align(self, depth_src, depth_target, valid_mask, image, **kwargs):
        depth_src = as_bchw_tensor(depth_src, 1)
        depth_target = as_bchw_tensor(depth_target, 1)
        valid_mask = as_bchw_tensor(valid_mask, 1)
        depth_target = scale_shift_linear(depth_target, depth_src, valid_mask)
        aligned = align_by_mlp(image, valid_mask, depth_target, depth_src, **kwargs)

        depth_numpy = aligned.squeeze().float().cpu().numpy()
        blur_bilateral = cv2.bilateralFilter(depth_numpy, 5, 140, 140)
        blur_gaussian = cv2.GaussianBlur(blur_bilateral, (5, 5), 0)
        blur_gaussian = torch.from_numpy(blur_gaussian).to(aligned)
        return blur_gaussian.unsqueeze(0)