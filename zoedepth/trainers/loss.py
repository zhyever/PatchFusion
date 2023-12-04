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

# File author: Shariq Farooq Bhat

# This file may include modifications from author Zhenyu Li

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

from torch.autograd import Variable
from math import exp

import matplotlib.pyplot as plt
KEY_OUTPUT = 'metric_depth'
# import kornia
import copy

def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        hack_input = input

        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            if input.numel() == 0:
                loss = torch.mean(hack_input) * 0
                if not return_interpolated:
                    return loss
                return loss, intr_input
        
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


# class GradL1Loss(nn.Module):
#     """Gradient loss"""
#     def __init__(self):
#         super(GradL1Loss, self).__init__()
#         self.name = 'GradL1'

#     def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
#         input = extract_key(input, KEY_OUTPUT)
#         if input.shape[-1] != target.shape[-1] and interpolate:
#             input = nn.functional.interpolate(
#                 input, target.shape[-2:], mode='bilinear', align_corners=True)
#             intr_input = input
#         else:
#             intr_input = input

#         grad_gt = grad(target)
#         grad_pred = grad(input)
#         mask_g = grad_mask(mask)

#         loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
#         loss = loss + \
#             nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
#         if not return_interpolated:
#             return loss
#         return loss, intr_input


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    
    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin




    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input
    



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input


class BudgetConstraint(nn.Module):
    """
    Given budget constraint to reduce expected inference FLOPs in the Dynamic Network.
    """
    def __init__(self, loss_mu, flops_all, warm_up=True):
        super().__init__()
        self.loss_mu = loss_mu
        self.flops_all = flops_all
        self.warm_up = warm_up

    def forward(self, flops_expt, warm_up_rate=1.0):
        if self.warm_up:
            warm_up_rate = min(1.0, warm_up_rate)
        else:
            warm_up_rate = 1.0
        losses =  warm_up_rate * ((flops_expt / self.flops_all - self.loss_mu)**2)
        return losses


if __name__ == '__main__':
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32)*10, torch.rand(4, 1, 26, 32)*10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))



class HistogramMatchingLoss(nn.Module):
    def __init__(self, min_depth, max_depth, bins=512):
        super(HistogramMatchingLoss, self).__init__()
        self.name = 'HistogramMatchingLoss'
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins = bins

    def forward(self, input, target, mask, interpolate=True):
        if input.shape[-1] != mask.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, mask.shape[-2:], mode='bilinear', align_corners=True)
        
        if target.shape[-1] != mask.shape[-1] and interpolate:
            target = nn.functional.interpolate(
                target, mask.shape[-2:], mode='bilinear', align_corners=True)

        input[~mask] = 0
        target[~mask] = 0


        pred_hist = torch.histc(input, bins=self.bins, min=self.min_depth, max=self.max_depth)
        gt_hist = torch.histc(target, bins=self.bins, min=self.min_depth, max=self.max_depth)

        pred_hist /= pred_hist.sum(dim=0, keepdim=True)
        gt_hist /= gt_hist.sum(dim=0, keepdim=True)

        # print(pred_hist.shape)
        # print(pred_hist)
        # _pred_hist = pred_hist.detach().cpu().numpy()
        # _gt_hist = gt_hist.detach().cpu().numpy()
        # plt.subplot(2, 1, 1)
        # plt.bar(range(len(_pred_hist)), _pred_hist)
        # plt.subplot(2, 1, 2)
        # plt.bar(range(len(_gt_hist)), _gt_hist)
        # plt.savefig('./debug_scale.png')

        # Compute cumulative histograms (CDF)
        cdf_pred = torch.cumsum(pred_hist, dim=0)
        cdf_gt = torch.cumsum(gt_hist, dim=0)

        # Compute Earth Mover's Distance (EMD) between the CDFs
        loss = torch.mean(torch.abs(cdf_pred - cdf_gt))
        # loss = torch.mean(torch.sqrt((pred_hist - gt_hist)**2))
        # loss = F.kl_div(torch.log(pred_hist + 1e-10), gt_hist, reduction='mean')
        
        return loss




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask, interpolate=True):
        if img1.shape[-1] != mask.shape[-1] and interpolate:
            img1 = nn.functional.interpolate(
                img1, mask.shape[-2:], mode='bilinear', align_corners=True)
        
        if img2.shape[-1] != mask.shape[-1] and interpolate:
            img2 = nn.functional.interpolate(
                img2, mask.shape[-2:], mode='bilinear', align_corners=True)

        img1[~mask] = 0
        img2[~mask] = 0

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        loss = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return loss

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
        
class ConsistencyLoss(nn.Module):
    def __init__(self, target, focus_flatten=False, wp=1) -> None:
        super().__init__()
        self.name = 'Consistency'
        self.target = target
        self.mode = 'no-resize'
        # self.mode = 'resize'
        self.focus_flatten = focus_flatten
        self.wp = wp

    def gradient_y(self, img):
        # gy = torch.cat([F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).to(img.device), padding=1) for i in range(img.shape[1])], 1)
        gy = F.conv2d(img, torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).to(img.device), padding=1)
        return gy

    def gradient_x(self, img):
        # gx = torch.cat([F.conv2d(img[:, i, :, :].unsqueeze(0), torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).to(img.device), padding=1) for i in range(img.shape[1])], 1)
        gx = F.conv2d(img, torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).to(img.device), padding=1)
        return gx

    def forward(self, depth_preds, shifts, mask, temp_features, pred_f=None):

        common_area_1_list = []
        common_area_2_list = []

        if self.focus_flatten:
            # only consider flatten place
            grad = kornia.filters.spatial_gradient(pred_f.detach())
            grad_x, grad_y = grad[:, :, 0, :, :], grad[:, :, 1, :, :]
            grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_ext = grad > 0.05 # over 5cm
            grad_ext = grad_ext.float()
            grad_blur = kornia.filters.gaussian_blur2d(grad_ext, (11, 11), (3, 3))
            grad_ext = grad_blur > 0 # over 5cm
            grad_ext = grad_blur == 0 
            mask = torch.logical_and(mask, grad_ext)


        if self.target == "mix":
            ## for feature
            bs, c, h, w = depth_preds.shape
            split_depth = torch.split(depth_preds, bs//2, dim=0)
            split_mask = torch.split(F.interpolate(mask.float(), (384, 512)).bool(), bs//2, dim=0)

            feat_ori_list = []
            feat_shift_list = []
            multi_level_mask = []

            for idx, feature in enumerate(temp_features): # multi-level
                split_feat = torch.split(feature, bs//2, dim=0)

                _, _, h, w = split_feat[0].shape
                feat_ori_list.append(split_feat[0])
                feat_shift_list.append(split_feat[1])

                mask_ori_cur_scale = F.interpolate(split_mask[0].float(), (h, w)).bool()
                multi_level_mask.append(mask_ori_cur_scale)

            for idx_out, (feat_ori_cur_level, feat_shift_cur_level, mask_ori_cur_level) in enumerate(zip(feat_ori_list, feat_shift_list, multi_level_mask)): # iter multi-scale
                scale_factor = 2 ** (5 - idx_out)
                _, _, cur_scale_h, cur_scale_w = feat_ori_cur_level.shape
                scale_factor = int(384 / cur_scale_h)

                for idx_in, (feat_ori, feat_shift, mask_ori, shift_bs) in enumerate(zip(feat_ori_cur_level, feat_shift_cur_level, mask_ori_cur_level, shifts)): # iter bs (paired feat)
                    c, _, _ = feat_ori.shape
                    mask_ori = mask_ori.repeat(c, 1, 1)
                    shift_h, shift_w = int(shift_bs[0] * (384/540) / scale_factor), int(shift_bs[1]* (512/960) / scale_factor)

                    if shift_h >= 0 and shift_w >= 0:
                        common_area_1 = feat_ori[:, shift_h:, shift_w:]
                        common_area_2 = feat_shift[:, :-shift_h, :-shift_w]
                        mask_common = mask_ori[:, shift_h:, shift_w:]       
                    elif shift_h >= 0 and shift_w <= 0:
                        common_area_1 = feat_ori[:, shift_h:, :-abs(shift_w)]
                        common_area_2 = feat_shift[:, :-shift_h, abs(shift_w):]
                        mask_common = mask_ori[:, shift_h:, :-abs(shift_w)]
                    elif shift_h <= 0 and shift_w <= 0:
                        common_area_1 = feat_ori[:, :-abs(shift_h), :-abs(shift_w)]
                        common_area_2 = feat_shift[:, abs(shift_h):, abs(shift_w):]
                        mask_common = mask_ori[:, :-abs(shift_h), :-abs(shift_w)]
                    elif shift_h <= 0 and shift_w >= 0:
                        common_area_1 = feat_ori[:, :-abs(shift_h):, shift_w:]
                        common_area_2 = feat_shift[:, abs(shift_h):, :-shift_w]
                        mask_common = mask_ori[:, :-abs(shift_h):, shift_w:]
                    else:
                        print("can you really reach here?")

                    common_area_masked_1 = common_area_1[mask_common].flatten()
                    common_area_masked_2 = common_area_2[mask_common].flatten()
                    common_area_1_list.append(common_area_masked_1)
                    common_area_2_list.append(common_area_masked_2)

            common_area_1 = torch.cat(common_area_1_list)
            common_area_2 = torch.cat(common_area_2_list)
            if common_area_1.numel() == 0 or common_area_2.numel() == 0:
                consistency_loss = torch.Tensor([0]).squeeze()
            else:
                consistency_loss = F.mse_loss(common_area_1, common_area_2)
            consistency_loss_feat = consistency_loss

            
            common_area_1_list = []
            common_area_2_list = []

            ## for pred
            bs, c, h, w = depth_preds.shape
            split_depth = torch.split(depth_preds, bs//2, dim=0)
            split_mask = torch.split(mask, bs//2, dim=0)
        
            for shift, depth_ori, depth_shift, mask_ori, mask_shift in zip(shifts, split_depth[0], split_depth[1], split_mask[0], split_mask[1]):
                shift_h, shift_w = shift[0], shift[1]
                if shift_h >= 0 and shift_w >= 0:
                    common_area_1 = depth_ori[:, shift_h:, shift_w:]
                    common_area_2 = depth_shift[:, :-shift_h, :-shift_w]
                    mask_common = mask_ori[:, shift_h:, shift_w:]
                    # mask_debug = mask_shift[:, :-shift_h, :-shift_w]
                elif shift_h >= 0 and shift_w <= 0:
                    common_area_1 = depth_ori[:, shift_h:, :-abs(shift_w)]
                    common_area_2 = depth_shift[:, :-shift_h, abs(shift_w):]
                    mask_common = mask_ori[:, shift_h:, :-abs(shift_w)]
                    # mask_debug = mask_shift[:, :-shift_h, abs(shift_w):]
                elif shift_h <= 0 and shift_w <= 0:
                    common_area_1 = depth_ori[:, :-abs(shift_h), :-abs(shift_w)]
                    common_area_2 = depth_shift[:, abs(shift_h):, abs(shift_w):]
                    mask_common = mask_ori[:, :-abs(shift_h), :-abs(shift_w)]
                    # mask_debug = mask_shift[:, abs(shift_h):, abs(shift_w):]
                elif shift_h <= 0 and shift_w >= 0:
                    common_area_1 = depth_ori[:, :-abs(shift_h):, shift_w:]
                    common_area_2 = depth_shift[:, abs(shift_h):, :-shift_w]
                    mask_common = mask_ori[:, :-abs(shift_h):, shift_w:]
                    # mask_debug = mask_shift[:, abs(shift_h):, :-shift_w]
                else:
                    print("can you really reach here?")
            
                common_area_1 = common_area_1[mask_common].flatten()
                common_area_2 = common_area_2[mask_common].flatten()
                common_area_1_list.append(common_area_1)
                common_area_2_list.append(common_area_2)

            common_area_1 = torch.cat(common_area_1_list)
            common_area_2 = torch.cat(common_area_2_list)
            if common_area_1.numel() == 0 or common_area_2.numel() == 0:
                consistency_loss = torch.Tensor([0]).squeeze()
            else:
                # pred_hist = torch.histc(common_area_1, bins=512, min=0, max=80)
                # gt_hist = torch.histc(common_area_2, bins=512, min=0, max=80)

                # pred_hist /= pred_hist.sum(dim=0, keepdim=True)
                # gt_hist /= gt_hist.sum(dim=0, keepdim=True)

                # # Compute cumulative histograms (CDF)
                # cdf_pred = torch.cumsum(pred_hist, dim=0)
                # cdf_gt = torch.cumsum(gt_hist, dim=0)

                # # Compute Earth Mover's Distance (EMD) between the CDFs
                # consistency_loss = torch.mean(torch.abs(cdf_pred - cdf_gt))
                consistency_loss = F.mse_loss(common_area_1, common_area_2) 
            consistency_loss_pred = consistency_loss

            consistency_loss = consistency_loss_pred * self.wp + consistency_loss_feat
            return consistency_loss
    
        elif 'feat' in self.target:
            if self.mode == 'resize':
                bs, c, h, w = depth_preds.shape
                split_depth = torch.split(depth_preds, bs//2, dim=0)
                split_mask = torch.split(mask, bs//2, dim=0)
                
                feat_ori_list = []
                feat_shift_list = []

                for idx, feature in enumerate(temp_features): # multi-level
                    if idx < 4:
                        continue
                    
                    split_feat = torch.split(feature, bs//2, dim=0)
                    f = F.interpolate(split_feat[0], (h, w), mode='bilinear', align_corners=True)
                    feat_ori_list.append(f)
                    f = F.interpolate(split_feat[1], (h, w), mode='bilinear', align_corners=True)
                    feat_shift_list.append(f)


                for idx_out, (feat_ori_cur_level, feat_shift_cur_level) in enumerate(zip(feat_ori_list, feat_shift_list)): # iter multi-scale
                    scale_factor = 2 ** (5 - idx_out)

                    for idx_in, (feat_ori, feat_shift, mask_ori, shift_bs) in enumerate(zip(feat_ori_cur_level, feat_shift_cur_level, split_mask[0], shifts)): # iter bs (paired feat)
                        c, h, w = feat_ori.shape
                        mask_ori = mask_ori.repeat(c, 1, 1)
                        shift_h, shift_w = shift_bs[0], shift_bs[1]

                        if shift_h >= 0 and shift_w >= 0:
                            common_area_1 = feat_ori[:, shift_h:, shift_w:]
                            common_area_2 = feat_shift[:, :-shift_h, :-shift_w]
                            mask_common = mask_ori[:, shift_h:, shift_w:]       
                        elif shift_h >= 0 and shift_w <= 0:
                            common_area_1 = feat_ori[:, shift_h:, :-abs(shift_w)]
                            common_area_2 = feat_shift[:, :-shift_h, abs(shift_w):]
                            mask_common = mask_ori[:, shift_h:, :-abs(shift_w)]
                        elif shift_h <= 0 and shift_w <= 0:
                            common_area_1 = feat_ori[:, :-abs(shift_h), :-abs(shift_w)]
                            common_area_2 = feat_shift[:, abs(shift_h):, abs(shift_w):]
                            mask_common = mask_ori[:, :-abs(shift_h), :-abs(shift_w)]
                        elif shift_h <= 0 and shift_w >= 0:
                            common_area_1 = feat_ori[:, :-abs(shift_h):, shift_w:]
                            common_area_2 = feat_shift[:, abs(shift_h):, :-shift_w]
                            mask_common = mask_ori[:, :-abs(shift_h):, shift_w:]
                        else:
                            print("can you really reach here?")

                        common_area_masked_1 = common_area_1[mask_common].flatten()
                        common_area_masked_2 = common_area_2[mask_common].flatten()
                        # common_area_masked_1 = common_area_1.flatten()
                        # common_area_masked_2 = common_area_2.flatten()
                        common_area_1_list.append(common_area_masked_1)
                        common_area_2_list.append(common_area_masked_2)

                common_area_1 = torch.cat(common_area_1_list)
                common_area_2 = torch.cat(common_area_2_list)
                if common_area_1.numel() == 0 or common_area_2.numel() == 0:
                    consistency_loss = torch.Tensor([0]).squeeze()
                else:
                    consistency_loss = F.mse_loss(common_area_1, common_area_2)

                return consistency_loss
            

            else:
                bs, c, h, w = depth_preds.shape
                split_depth = torch.split(depth_preds, bs//2, dim=0)
                mask = F.interpolate(mask.float(), (384, 512)).bool() # back to 384, 512
                split_mask = torch.split(mask, bs//2, dim=0)

                feat_ori_list = []
                feat_shift_list = []
                multi_level_mask = []

                for idx, feature in enumerate(temp_features): # multi-level
                    split_feat = torch.split(feature, bs//2, dim=0)

                    _, _, h, w = split_feat[0].shape
                    feat_ori_list.append(split_feat[0])
                    feat_shift_list.append(split_feat[1])

                    mask_ori_cur_scale = F.interpolate(split_mask[0].float(), (h, w)).bool()
                    multi_level_mask.append(mask_ori_cur_scale)

                for idx_out, (feat_ori_cur_level, feat_shift_cur_level, mask_ori_cur_level) in enumerate(zip(feat_ori_list, feat_shift_list, multi_level_mask)): # iter multi-scale
                    scale_factor = 2 ** (5 - idx_out)
                    _, _, cur_scale_h, cur_scale_w = feat_ori_cur_level.shape
                    scale_factor = int(384 / cur_scale_h)

                    for idx_in, (feat_ori, feat_shift, mask_ori, shift_bs) in enumerate(zip(feat_ori_cur_level, feat_shift_cur_level, mask_ori_cur_level, shifts)): # iter bs (paired feat)
                        c, _, _ = feat_ori.shape
                        mask_ori = mask_ori.repeat(c, 1, 1)
                        shift_h, shift_w = int(shift_bs[0] * (384/540) / scale_factor), int(shift_bs[1]* (512/960) / scale_factor)

                        if shift_h >= 0 and shift_w >= 0:
                            common_area_1 = feat_ori[:, shift_h:, shift_w:]
                            common_area_2 = feat_shift[:, :-shift_h, :-shift_w]
                            mask_common = mask_ori[:, shift_h:, shift_w:]       
                        elif shift_h >= 0 and shift_w <= 0:
                            common_area_1 = feat_ori[:, shift_h:, :-abs(shift_w)]
                            common_area_2 = feat_shift[:, :-shift_h, abs(shift_w):]
                            mask_common = mask_ori[:, shift_h:, :-abs(shift_w)]
                        elif shift_h <= 0 and shift_w <= 0:
                            common_area_1 = feat_ori[:, :-abs(shift_h), :-abs(shift_w)]
                            common_area_2 = feat_shift[:, abs(shift_h):, abs(shift_w):]
                            mask_common = mask_ori[:, :-abs(shift_h), :-abs(shift_w)]
                        elif shift_h <= 0 and shift_w >= 0:
                            common_area_1 = feat_ori[:, :-abs(shift_h):, shift_w:]
                            common_area_2 = feat_shift[:, abs(shift_h):, :-shift_w]
                            mask_common = mask_ori[:, :-abs(shift_h):, shift_w:]
                        else:
                            print("can you really reach here?")

                        common_area_masked_1 = common_area_1[mask_common].flatten()
                        common_area_masked_2 = common_area_2[mask_common].flatten()
                        common_area_1_list.append(common_area_masked_1)
                        common_area_2_list.append(common_area_masked_2)

                common_area_1 = torch.cat(common_area_1_list)
                common_area_2 = torch.cat(common_area_2_list)
                if common_area_1.numel() == 0 or common_area_2.numel() == 0:
                    consistency_loss = torch.Tensor([0]).squeeze()
                else:
                    consistency_loss = F.mse_loss(common_area_1, common_area_2)
                return consistency_loss
        
        elif self.target == 'pred':
            bs, c, h, w = depth_preds.shape
            split_depth = torch.split(depth_preds, bs//2, dim=0)
            split_mask = torch.split(mask, bs//2, dim=0)
        
            for shift, depth_ori, depth_shift, mask_ori, mask_shift in zip(shifts, split_depth[0], split_depth[1], split_mask[0], split_mask[1]):
                shift_h, shift_w = shift[0], shift[1]
                if shift_h >= 0 and shift_w >= 0:
                    common_area_1 = depth_ori[:, shift_h:, shift_w:]
                    common_area_2 = depth_shift[:, :-shift_h, :-shift_w]
                    mask_common = mask_ori[:, shift_h:, shift_w:]
                    # mask_debug = mask_shift[:, :-shift_h, :-shift_w]
                elif shift_h >= 0 and shift_w <= 0:
                    common_area_1 = depth_ori[:, shift_h:, :-abs(shift_w)]
                    common_area_2 = depth_shift[:, :-shift_h, abs(shift_w):]
                    mask_common = mask_ori[:, shift_h:, :-abs(shift_w)]
                    # mask_debug = mask_shift[:, :-shift_h, abs(shift_w):]
                elif shift_h <= 0 and shift_w <= 0:
                    common_area_1 = depth_ori[:, :-abs(shift_h), :-abs(shift_w)]
                    common_area_2 = depth_shift[:, abs(shift_h):, abs(shift_w):]
                    mask_common = mask_ori[:, :-abs(shift_h), :-abs(shift_w)]
                    # mask_debug = mask_shift[:, abs(shift_h):, abs(shift_w):]
                elif shift_h <= 0 and shift_w >= 0:
                    common_area_1 = depth_ori[:, :-abs(shift_h):, shift_w:]
                    common_area_2 = depth_shift[:, abs(shift_h):, :-shift_w]
                    mask_common = mask_ori[:, :-abs(shift_h):, shift_w:]
                    # mask_debug = mask_shift[:, abs(shift_h):, :-shift_w]
                else:
                    print("can you really reach here?")
            
                common_area_1 = common_area_1[mask_common].flatten()
                common_area_2 = common_area_2[mask_common].flatten()
                common_area_1_list.append(common_area_1)
                common_area_2_list.append(common_area_2)

            common_area_1 = torch.cat(common_area_1_list)
            common_area_2 = torch.cat(common_area_2_list)
            if common_area_1.numel() == 0 or common_area_2.numel() == 0:
                consistency_loss = torch.Tensor([0]).squeeze()
            else:
                # pred_hist = torch.histc(common_area_1, bins=512, min=0, max=80)
                # gt_hist = torch.histc(common_area_2, bins=512, min=0, max=80)

                # pred_hist /= pred_hist.sum(dim=0, keepdim=True)
                # gt_hist /= gt_hist.sum(dim=0, keepdim=True)

                # # Compute cumulative histograms (CDF)
                # cdf_pred = torch.cumsum(pred_hist, dim=0)
                # cdf_gt = torch.cumsum(gt_hist, dim=0)

                # # Compute Earth Mover's Distance (EMD) between the CDFs
                # consistency_loss = torch.mean(torch.abs(cdf_pred - cdf_gt))
                consistency_loss = F.mse_loss(common_area_1, common_area_2)
            
            return consistency_loss
            
        else:
            raise NotImplementedError