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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from typing import List, Tuple
from PIL import Image
# from models.monodepth.zoedepth import ZoeDepthLora
# from zoedepth.utils.align.loss import SILogLoss, gradl1_loss, edge_aware_smoothness_per_pixel, ssim_loss
from .loss import *
import cv2
from zoedepth.trainers.loss import *
# from utils.misc import *



@torch.no_grad()
def scale_shift_linear(rendered_depth, predicted_depth, mask, fuse=True, return_params=False):
    """
    Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
    Formally, solves the following objective:

    min     || (d * a + b) - d_hat ||
    a, b

    where d = 1 / predicted_depth, d_hat = 1 / rendered_depth

    :param rendered_depth: torch.Tensor (H, W)
    :param predicted_depth:  torch.Tensor (H, W)
    :param mask: torch.Tensor (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
    :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth

    :return: scale/shift corrected depth
    """
    if mask.sum() == 0:
        return predicted_depth

    # rendered_disparity = 1 / rendered_depth[mask].unsqueeze(-1)
    # predicted_disparity = 1 / predicted_depth[mask].unsqueeze(-1)

    rendered_disparity = rendered_depth[mask].unsqueeze(-1)
    predicted_disparity = predicted_depth[mask].unsqueeze(-1)

    X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ rendered_disparity
    AB = XTX_inv @ XTY

    if return_params:
        return AB

    fixed_disparity = (predicted_depth) * AB[0] + AB[1]
    fixed_depth = fixed_disparity

    if fuse:
        fused_depth = torch.where(mask, rendered_depth, fixed_depth)
        return fused_depth
    else:
        return fixed_depth
    

def np_scale_shift_linear(rendered_depth: np.ndarray, predicted_depth: np.ndarray, mask: np.ndarray, fuse: bool=True):
    """
    Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
    Formally, solves the following objective:

    min     || (d * a + b) - d_hat ||
    a, b

    where d = predicted_depth, d_hat = rendered_depth

    :param rendered_depth: np.ndarray (H, W)
    :param predicted_depth:  np.ndarray (H, W)
    :param mask: np.ndarray (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
    :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth

    :return: scale/shift corrected depth
    """
    if mask.sum() == 0:
        return predicted_depth

    # rendered_disparity = 1 / rendered_depth[mask].reshape(-1, 1)
    # predicted_disparity = 1 / predicted_depth[mask].reshape(-1, 1)

    rendered_disparity = rendered_depth[mask].reshape(-1, 1)
    predicted_disparity = predicted_depth[mask].reshape(-1, 1)

    X = np.concatenate([predicted_disparity, np.ones_like(predicted_disparity)], axis=1)
    XTX_inv = np.linalg.inv(X.T @ X)
    XTY = X.T @ rendered_disparity
    AB = XTX_inv @ XTY

    fixed_disparity = (predicted_depth) * AB[0] + AB[1]
    fixed_depth = fixed_disparity

    if fuse:
        fused_depth = np.where(mask, rendered_depth, fixed_depth)
        return fused_depth
    else:
        return fixed_depth


@torch.no_grad()
def apply_depth_smoothing(depth, mask):

    def dilate(x, k=3):
        x = as_bchw_tensor(x.float(), 1)
        x = torch.nn.functional.conv2d(x.float(),
            torch.ones(1, 1, k, k).to(x.device),
            padding="same"
        )
        return x.squeeze() > 0

    def sobel(x):
        flipped_sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).to(x.device)
        flipped_sobel_x = torch.stack([flipped_sobel_x, flipped_sobel_x.t()]).unsqueeze(1)

        x_pad = torch.nn.functional.pad(x.float(), (1, 1, 1, 1), mode="replicate")

        x = torch.nn.functional.conv2d(
            x_pad,
            flipped_sobel_x,
            padding="valid"
        )
        dx, dy = x.unbind(dim=-3)
        # return torch.sqrt(dx**2 + dy**2).squeeze()
        # new content is created mostly in x direction, sharp edges in y direction are wanted (e.g. table --> wall)
        return dx

    depth = as_bchw_tensor(depth, 1)
    mask = as_bchw_tensor(mask, 1).float()

    edges = sobel(mask)
    dilated_edges = dilate(edges, k=21)

    depth_numpy = depth.squeeze().float().cpu().numpy()
    blur_bilateral = cv2.bilateralFilter(depth_numpy, 5, 140, 140)
    blur_gaussian = cv2.GaussianBlur(blur_bilateral, (5, 5), 0)
    blur_gaussian = torch.from_numpy(blur_gaussian).to(depth)
    # print("blur_gaussian", blur_gaussian.shape)
    # plt.imshow(blur_gaussian.cpu().squeeze().numpy())
    # plt.title("depth smoothed whole")
    # plt.show()
    depth_smooth = torch.where(dilated_edges, blur_gaussian, depth)
    return depth_smooth

def get_dilated_only_mask(mask: torch.Tensor, k=7):
    x = as_bchw_tensor(mask.float(), 1)
    x = torch.nn.functional.conv2d(x, torch.ones(1, 1, k, k).to(mask.device),padding="same")
    dilated = x.squeeze() > 0
    dilated_only = dilated ^ mask 
    return dilated_only

def get_boundary_mask(mask: torch.Tensor, k=7):
    return get_dilated_only_mask(mask, k=k) | get_dilated_only_mask(~mask, k=k)


@torch.no_grad()
def ss_align_and_blur(rendered_depth: torch.Tensor, predicted_depth: torch.Tensor, mask: torch.Tensor, fuse: bool=True):
    aligned = scale_shift_linear(rendered_depth, predicted_depth, mask, fuse=fuse)
    aligned = apply_depth_smoothing(aligned, mask)
    return aligned


def np_ss_align_and_blur(rendered_depth: np.ndarray, predicted_depth: np.ndarray, mask: np.ndarray, fuse: bool=True):
    aligned = np_scale_shift_linear(rendered_depth, predicted_depth, mask, fuse=fuse)
    aligned = apply_depth_smoothing(aligned, mask).cpu().numpy()
    return aligned



def stitch(depth_src: torch.Tensor, depth_target: torch.Tensor, mask_src: torch.Tensor, smoothen=True, device='cuda:0'):
    depth_src = as_bchw_tensor(depth_src, 1, device=device)
    depth_target = as_bchw_tensor(depth_target, 1, device=device)
    mask_src = as_bchw_tensor(mask_src, 1, device=device)

    stitched = depth_src * mask_src.float() + depth_target * (~mask_src).float()
    # plt.imshow(stitched.cpu().squeeze().numpy())
    # plt.title("stitched before smoothing")
    # plt.show()
    # apply smoothing
    if smoothen:
        stitched = apply_depth_smoothing(stitched, mask_src).squeeze().float()
    return stitched









def smoothness_loss(depth, mask=None):
    depth_grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    depth_grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    if mask is not None:
        return torch.mean(depth_grad_x[mask[:, :, :, :-1]]) + torch.mean(depth_grad_y[mask[:, :, :-1, :]])
    return torch.mean(depth_grad_x) + torch.mean(depth_grad_y)

import torch.optim as optim
from torch.optim import lr_scheduler
def finetune_on_sample(model, image_pil, target_depth, mask=None, 
                       iters=10, lr=0.1, beta=0.5, w_boundary_grad=1, w_grad=0.1, gamma=0.99):
    model.train()
    model_device = next(model.parameters()).device
    x = as_bchw_tensor(image_pil, 3, device=model_device)
    target_depth = as_bchw_tensor(target_depth, 1, device=model_device)
    if mask is None:
        mask = target_depth > 0
    elif (not isinstance(mask, torch.Tensor)) or mask.shape != target_depth.shape:
        mask = as_bchw_tensor(mask, 1, device=model_device).to(torch.bool)
        
    
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=iters, epochs=1)
    # main_loss = nn.L1Loss()
    main_loss = SILogLoss(beta=beta)

    orig_y = model.infer(x, with_flip_aug=False).detach()

    # scale, shift = scale_shift_linear(target_depth, orig_y, mask, return_params=True)

    gl1 = gradl1_loss
    pbar = tqdm(range(iters), desc="Finetuning on sample")
    for i in pbar:
        optimizer.zero_grad()
        y = model.infer(x, with_flip_aug=False)
        # y = y * scale + shift
        stitched = y * (~mask).float() + (target_depth * (mask).float()).detach()
        # loss = F.mse_loss(y[mask], target_depth[mask])
        loss_si = main_loss(y[mask], target_depth[mask])
        # loss = loss_si \
        #     + wgrad * ( gl1(y, stitched) \
        #                + 2*gl1(y, orig_y) ) \
        #         + wboundary_smoothness * smoothness_loss(y, mask=get_boundary_mask(mask))
        loss_grad = gl1(y, orig_y)

        bmask = get_boundary_mask(mask)
        loss_boundary_grad = laplacian_matching_loss(stitched, orig_y, bmask)
        loss = loss_si + w_boundary_grad * loss_boundary_grad + w_grad * loss_grad

        # check if loss is nan
        if torch.isnan(loss):
            print("Loss is nan, breaking")
            break
        loss.backward()

        optimizer.step()
        scheduler.step()
        # history.append(loss.item())

        pbar.set_postfix(loss=loss.item(), si=loss_si.item())
    model.eval()
    return model, history

# def align_by_finetuning_lora(model: ZoeDepthLora, image, target_depth, mask=None, iters=10, lr=0.1, gamma=0.99, **kwargs):
#     # model.reset_lora()
#     model.set_only_lora_trainable()
#     model, history = finetune_on_sample(model, image, target_depth, mask=mask, iters=iters, lr=lr, gamma=gamma)
#     aligned_depth = model.infer(as_bchw_tensor(image, 3, device=next(model.parameters()).device))
#     return dict(model=model, history=history, aligned_depth=aligned_depth)






import torch.nn as nn
import torch.nn.functional as F
# from utils.misc import as_bchw_tensor

def as_bchw_tensor(input_tensor, num, device):
    input_tensor = torch.tensor(input_tensor).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    return input_tensor

def optimize_depth_deformation(rendered_depth, pred_depth, mask, h=10, w=10, iters=100, init_lr=0.1, gamma=0.996,
                                init_deformation=None,
                                device='cuda:0'):
    rendered_depth = as_bchw_tensor(rendered_depth, 1, device=device)
    pred_depth = as_bchw_tensor(pred_depth, 1, device=device)
    mask = as_bchw_tensor(mask, 1, device=device).to(torch.bool)
    # initialize a grid of scalar values (with zeros) that will be optimized
    # to deform the depth map
    if init_deformation is None:
        deformation = torch.zeros((1,1,h,w), requires_grad=True, device=device)
    else:
        deformation = init_deformation
        deformation.requires_grad = True
        assert deformation.shape == (1,1,h,w)

    optimizer = torch.optim.Adam([deformation], lr=init_lr)
    # exponential LR schedule
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # optimize the deformation
    history = []
    grad_loss = GradL1Loss()
    for i in tqdm(range(iters)):
        scalar_deformation = torch.exp(deformation)
        scalar_deformation = F.interpolate(scalar_deformation, size=pred_depth.shape[-2:], mode='bilinear', align_corners=True)
        adjusted_depth = pred_depth * scalar_deformation
        loss = F.mse_loss(adjusted_depth[mask], rendered_depth[mask], reduction='none')
        loss_g = grad_loss(adjusted_depth, rendered_depth, mask)
        loss = loss.mean() + 0.1*loss_g
        # loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:
            history.append(loss.item())
    
    scalar_deformation = torch.exp(deformation)
    scalar_deformation = F.interpolate(scalar_deformation, size=pred_depth.shape[-2:], mode='bilinear', align_corners=True)
    adjusted_depth = pred_depth * scalar_deformation
    # return dict(aligned_depth=adjusted_depth.detach().cpu().numpy().squeeze(), 
    #             history=history, 
    #             deformation=deformation)
    return adjusted_depth.detach().cpu().squeeze()

def stage_wise_optimization(rendered_depth, pred_depth, mask,
                            stages=[(4,4), (8,8), (16,16), (32,32)], 
                            iters=100, init_lr=0.1, gamma=0.996, device='cuda:1'):
    
    h_init, w_init = stages[0]
    init_deformation = torch.zeros((1,1,h_init,w_init), device=device)
    result = optimize_depth_deformation(rendered_depth, pred_depth, mask, h=h_init, w=w_init, iters=iters, init_lr=init_lr, gamma=gamma, init_deformation=init_deformation, device=device)
    init_deformation = result['deformation']
    history_stages = [result['history']]
    for h, w in stages[1:]:
        init_deformation = F.interpolate(init_deformation, size=(h,w), mode='bilinear', align_corners=True).detach()
        result = optimize_depth_deformation(rendered_depth, pred_depth, mask, h=h, w=w, iters=iters, init_lr=init_lr, gamma=gamma, init_deformation=init_deformation, device=device)
        init_deformation = result['deformation']
        history_stages.append(result['history'])
        init_lr *= gamma**2
    
    return dict(aligned_depth=result['aligned_depth'], history_stages=history_stages)
        




