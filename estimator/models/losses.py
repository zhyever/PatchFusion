import copy
import kornia

import torch
import torch.nn as nn
from mmengine import print_log
import torch.nn.functional as F
import random
import math

from estimator.registry import MODELS
from kornia.losses import dice_loss, focal_loss

@MODELS.register_module()
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15, **kwargs):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, min_depth, max_depth, additional_mask=None):
        _, _, h_i, w_i = input.shape
        _, _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)
        
        mask = torch.logical_and(target>min_depth, target<max_depth)
        
        if additional_mask is not None:
            mask_merge = torch.logical_and(mask, additional_mask)
            if torch.sum(mask_merge) >= h_i * w_i * 0.001:
                mask = mask_merge
            else:
                print_log("torch.sum(mask_merge) < h_i * w_i * 0.001, reduce to previous mask for stable training", logger='current')
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding nan", logger='current')
            return input * 0.0
        
        input = input[mask]
        target = target[mask]
        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)
        Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)
        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print_log("Nan SILog loss", logger='current')
            print_log("input: {}".format(input.shape), logger='current')
            print_log("target: {}".format(target.shape), logger='current')
            
            print_log("G: {}".format(torch.sum(torch.isnan(g))), logger='current')
            print_log("Input min: {} max: {}".format(torch.min(input), torch.max(input)), logger='current')
            print_log("Target min: {} max: {}".format(torch.min(target), torch.max(target)), logger='current')
            print_log("Dg: {}".format(torch.isnan(Dg)), logger='current')
            print_log("loss: {}".format(torch.isnan(loss)), logger='current')

        return loss


def get_grad_map(value):
    grad = kornia.filters.spatial_gradient(value)
    grad_xy = (grad[:,:,0,:,:] ** 2 + grad[:,:,1,:,:] ** 2) ** (1/2)
    return grad_xy

def get_grad_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    grad_error[grad_error>0.001] = 1.
    grad_error[invalid_mask] = 2. # filter invalid area
    grad_error[gt>10000] = 3.
    return grad_error.long().squeeze(dim=1)

def get_grad_value_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    error = ((gt - coarse_prediction) / gt).abs() 
    error[error>0.1] = 1.
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    error[grad_error>0.001] = 1.
    error[invalid_mask] = 2. # filter invalid area
    error[gt>10000] = 3.
    return error.long().squeeze(dim=1)

def get_incoherent_mask(gt, shape=(384, 512), min_depth=1e-3, max_depth=80):
    # incoherent
    ori_shpae = gt.shape[-2:]
    gt_lr = F.interpolate(gt, shape, mode='bilinear', align_corners=True)
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_recover = F.interpolate(gt_lr, ori_shpae, mode='bilinear', align_corners=True)
    residue = (gt - gt_recover).abs()
    
    gt_label = torch.zeros_like(gt)
    gt_label[residue >= 0.01] = 1. # set incoherent area as 1
    gt_label[invalid_mask] = 2. # filter invalid area
    gt_label[gt>10000] = 3.
    return gt_label.long().squeeze(dim=1)

def get_incoherent_grad_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    # incoherent
    ori_shpae = gt.shape[-2:]
    gt_lr = F.interpolate(gt, shape, mode='bilinear', align_corners=True)
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_recover = F.interpolate(gt_lr, ori_shpae, mode='bilinear', align_corners=True)
    residue = (gt - gt_recover).abs()
    # coarse_prediction = F.interpolate(coarse_prediction, gt.shape[-2:], mode='bilinear', align_corners=True)
    # error = (gt - coarse_prediction).abs()
    
    # grad error
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    
    bad_area_mask = torch.logical_or(residue>0.01, grad_error>0.001)
    gt_label = torch.zeros_like(gt)
    gt_label[bad_area_mask] = 1.
    gt_label[invalid_mask] = 2. # filter invalid area
    gt_label[gt>10000] = 3.
    return gt_label.long().squeeze(dim=1)

def get_incoherent_grad_value_error_mask(gt, coarse_prediction, shape=(384, 512), min_depth=1e-3, max_depth=80):
    # incoherent
    ori_shpae = gt.shape[-2:]
    gt_lr = F.interpolate(gt, shape, mode='bilinear', align_corners=True)
    invalid_mask = torch.logical_or(gt<=min_depth, gt>=max_depth)
    gt_recover = F.interpolate(gt_lr, ori_shpae, mode='bilinear', align_corners=True)
    residue = (gt - gt_recover).abs()
    
    # value error
    coarse_prediction = F.interpolate(coarse_prediction, gt.shape[-2:], mode='bilinear', align_corners=True)
    error = (gt - coarse_prediction).abs()
    bad_area_mask = torch.logical_or(residue>0.01, error>0.5)
    
    # grad error
    gt_grad = get_grad_map(gt)
    coarse_prediction_grad = get_grad_map(coarse_prediction)
    grad_error = ((gt_grad - coarse_prediction_grad) / gt).abs()
    bad_area_mask = torch.logical_or(grad_error, grad_error>0.001)
    
    gt_label = torch.zeros_like(gt)
    gt_label[bad_area_mask] = 1.
    gt_label[invalid_mask] = 2. # filter invalid area
    gt_label[gt>10000] = 3.
    return gt_label.long().squeeze(dim=1)

class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self,
                 p=1,
                 smooth=1,
                 reduction='mean'):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, probs, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # compute loss
        numer = torch.sum((probs*label), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + label.pow(self.p), dim=(2, 3))
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

@MODELS.register_module()
class EdgeClsLoss(nn.Module):
    """Error loss (pixel-wise)"""
    def __init__(self, focal_weight=0.5):
        super(EdgeClsLoss, self).__init__()
        self.name = 'Error'
        self.criterion_dice = GeneralizedSoftDiceLoss()
        self.criterion_bce = nn.BCELoss()
        self.focal_weight = focal_weight

    def forward(self, input, target):
        _, _, h_i, w_i = input.shape
        _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)

        target = target.long()
        dice = dice_loss(input, target)
        focal = focal_loss(input, target, alpha=self.focal_weight, reduction='mean')
        
        return dice, focal


@MODELS.register_module()
class ErrorLoss(nn.Module):
    """Error loss (pixel-wise)"""
    def __init__(self, loss_type, focal_weight):
        super(ErrorLoss, self).__init__()
        self.name = 'Error'
        self.criterion_dice = GeneralizedSoftDiceLoss()
        self.criterion_bce = nn.BCELoss()
        self.loss_type = loss_type
        self.focal_weight = focal_weight

    def forward(self, input, target, coarse_prediction, min_depth, max_depth):
        _, _, h_i, w_i = input.shape
        _, _, h_c, w_c = coarse_prediction.shape
        _, _, h_t, w_t = target.shape
        
        if h_i != h_t or w_i != w_t:
            input = F.interpolate(input, (h_t, w_t), mode='bilinear', align_corners=True)
        if h_c != h_t or w_c != w_t:
            coarse_prediction = F.interpolate(coarse_prediction, (h_t, w_t), mode='bilinear')
            
        
        if self.loss_type == 'incoh':
            gt_mask = get_incoherent_mask(target, shape=(h_i, w_i), min_depth=min_depth, max_depth=max_depth)
        elif self.loss_type == 'incoh+grad':
            gt_mask = get_incoherent_grad_error_mask(target, coarse_prediction, shape=(h_i, w_i), min_depth=min_depth, max_depth=max_depth)
        elif self.loss_type == 'incoh+grad+depth':
            gt_mask = get_incoherent_grad_value_error_mask(target, coarse_prediction, shape=(h_i, w_i), min_depth=min_depth, max_depth=max_depth)
        else:
            raise NotImplementedError
        

        dice = dice_loss(input, gt_mask)
        # focal = focal_loss(input, gt_mask, alpha=0.5, reduction='mean')
        focal = focal_loss(input, gt_mask, alpha=self.focal_weight, reduction='mean')
        
        return dice, focal, gt_mask


def ind2sub(idx, w):
    # r = idx // cols
    # c = idx - r * cols
    # return r, c
    row = idx // w
    col = idx % w
    return row, col

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx
import matplotlib.pyplot as plt

######################################################
# EdgeguidedRankingLoss
#####################################################
@MODELS.register_module()
class EdgeguidedRankingLoss(nn.Module):
    def __init__(
        self, 
        point_pairs=10000, 
        sigma=0.03, 
        alpha=1.0, 
        mask_value=-1e-8, 
        reweight_target=False, 
        only_missing_area=False,
        min_depth=-1e-3, 
        max_depth=80,
        missing_value=-99,
        random_direct=True):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        #self.regularization_loss = GradientLoss(scales=4)
        self.reweight_target = reweight_target
        self.only_missing_area = only_missing_area
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.missing_value = missing_value
        self.random_direct = random_direct
        
        self.idx = 0
        self.idx_inner = 0
        
    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas

    def edgeGuidedSampling(self, inputs, targets, edges_img, thetas_img, missing_mask, depth_gt, strict_mask):
        # find edges
        edges_max = edges_img.max()
        edges_mask = edges_img.ge(edges_max*0.1)
        edges_mask = torch.logical_and(edges_mask, strict_mask) # 1 edge, 2 strict mask
        
        if self.only_missing_area:
            # edges_mask = torch.logical_and(edges_mask, missing_mask) # base anchor: 1 missing values (0) and 2 edge masks and within 3 strict mask
            edges_mask = missing_mask
            
        edges_loc = edges_mask.nonzero()

        minlen = edges_loc.shape[0]
        
        if minlen == 0:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), 0

        # find anchor points (i.e, edge points)
        sample_num = self.point_pairs
        sample_index = torch.randint(0, minlen, (sample_num,), dtype=torch.long).cuda()
        sample_h, sample_w = edges_loc[sample_index, 0], edges_loc[sample_index, 1]
        theta_anchors = thetas_img[sample_h, sample_w] 
        
        sidx = edges_loc.shape[0] // 2
        # plt.figure()
        # plt.imshow(missing_mask.squeeze().cpu().numpy())
        # plt.savefig('nfs_scannet/debug_figs/debug_mask_{}.png'.format(self.idx))
        # plt.figure()
        # plt.imshow(missing_mask.squeeze().cpu().numpy())
        # circle = plt.Circle((sample_w[0], sample_h[0]), 0.5, color='r')
        # plt.gca().add_patch(circle)
        # plt.savefig('nfs_scannet/debug_figs/debug_anchor_{}.png'.format(self.idx))
        
        ## compute the coordinates of 4-points,  distances are from [-30, 30]
        distance_matrix = torch.randint(2, 31, (4,sample_num)).cuda()
        pos_or_neg = torch.ones(4, sample_num).cuda()
        pos_or_neg[:2,:] = -pos_or_neg[:2,:]
            
        distance_matrix = distance_matrix.float() * pos_or_neg
        
        p = random.random()
        # theta_anchors = theta_anchors + math.pi / 2
        # # Normalize the angle to be between -pi and pi
        # theta_anchors = (theta_anchors + math.pi) % (2 * math.pi) - math.pi
        # col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()
        # row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
        
        if self.random_direct:
            if p < 0.5:
                col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
                row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()
            else:
                theta_anchors = theta_anchors + math.pi / 2
                theta_anchors = (theta_anchors + math.pi) % (2 * math.pi) - math.pi
                col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()
                row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
        else:
            col = sample_w.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
            row = sample_h.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()

        # constrain 0=<c<=w, 0<=r<=h
        # Note: index should minus 1
        w, h = depth_gt.shape[-1], depth_gt.shape[-2]
        invalid_mask = (col<0) + (col>w-1) + (row<0) + (row>h-1)
        invalid_mask = torch.sum(invalid_mask, dim=0) > 0
        col = col[:, torch.logical_not(invalid_mask)]
        row = row[:, torch.logical_not(invalid_mask)]
    
        # only use one pair
        a = torch.stack([row[0, :], col[0, :]])
        b = torch.stack([row[1, :], col[1, :]])
        c = torch.stack([row[2, :], col[2, :]])
        d = torch.stack([row[3, :], col[3, :]])
        
        
        if self.only_missing_area:
            valid_check_a_strict = strict_mask[a[0, :], a[1, :]] == True
            valid_check_a_missing = missing_mask[a[0, :], a[1, :]] == True
            valid_check_b_strict = strict_mask[b[0, :], b[1, :]] == True
            valid_check_b_missing = missing_mask[b[0, :], b[1, :]] == True
            valid_check_c_strict = strict_mask[c[0, :], c[1, :]] == True
            valid_check_c_missing = missing_mask[c[0, :], c[1, :]] == True
            valid_check_d_strict = strict_mask[d[0, :], d[1, :]] == True
            valid_check_d_missing = missing_mask[d[0, :], d[1, :]] == True
            
            valid_mask_ab = torch.logical_not(torch.logical_and(valid_check_a_strict, valid_check_b_strict))
            valid_mask_bc = torch.logical_not(torch.logical_and(valid_check_b_strict, valid_check_c_strict))
            valid_mask_cd = torch.logical_not(torch.logical_and(valid_check_c_strict, valid_check_d_strict))

            valid_mask = torch.logical_and(valid_mask_ab, valid_mask_bc)
            valid_mask = torch.logical_and(valid_mask, valid_mask_cd)
            
            # valid_check_a_missing = missing_mask[a[0, :], a[1, :]] == True
            # valid_check_b_missing = missing_mask[b[0, :], b[1, :]] == True
            # valid_check_c_missing = missing_mask[c[0, :], c[1, :]] == True
            # valid_check_d_missing = missing_mask[d[0, :], d[1, :]] == True
            # valid_mask = torch.logical_and(valid_check_a_missing, valid_check_b_missing)
            # valid_mask = torch.logical_and(valid_mask, valid_check_c_missing)
            # valid_mask = torch.logical_and(valid_mask, valid_check_d_missing)
            a = a[:, valid_mask]
            b = b[:, valid_mask]   
            c = c[:, valid_mask]
            d = d[:, valid_mask]   
        
        if a.numel() == 0 or b.numel() == 0 or c.numel() == 0 or d.numel() == 0:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), 0
        
        # sidx = 0
        # plt.figure()
        # plt.imshow(depth_gt.squeeze().cpu().numpy())
        # circle = plt.Circle((a[1][sidx], a[0][sidx]), 3, color='r')
        # plt.gca().add_patch(circle)
        # circle = plt.Circle((b[1][sidx], b[0][sidx]), 3, color='r')
        # plt.gca().add_patch(circle)
        # plt.savefig('nfs_scannet/debug_figs/debug_{}_{}.png'.format(self.idx, self.idx_inner))
        # self.idx_inner += 1
        
        A = torch.cat((a,b,c), 1)
        B = torch.cat((b,c,d), 1)
        sumple_num = A.shape[1]
        
        inputs_A = inputs[A[0, :], A[1, :]]
        inputs_B = inputs[B[0, :], B[1, :]]
        targets_A = targets[A[0, :], A[1, :]]
        targets_B = targets[B[0, :], B[1, :]]

        return inputs_A, inputs_B, targets_A, targets_B, sumple_num


    def randomSampling(self, inputs, targets, missing_part, valid_part, sample_num):
        # Apply masks to get the valid indices for missing and valid parts
        missing_indices = torch.nonzero(missing_part.float()).squeeze()
        valid_indices = torch.nonzero(valid_part.float()).squeeze()

        # Ensure that we have enough points to sample from
        sample_num = min(sample_num, len(missing_indices), len(valid_indices))

        # Shuffle and sample indices from the missing and valid parts
        shuffle_missing_indices = torch.randperm(len(missing_indices))[:sample_num].cuda()
        shuffle_valid_indices = torch.randperm(len(valid_indices))[:sample_num].cuda()

        # Select the sampled points for inputs and targets based on the shuffled indices
        inputs_A = inputs[missing_indices[shuffle_missing_indices]]
        inputs_B = inputs[valid_indices[shuffle_valid_indices]]

        targets_A = targets[missing_indices[shuffle_missing_indices]]
        targets_B = targets[valid_indices[shuffle_valid_indices]]
        return inputs_A, inputs_B, targets_A, targets_B, sample_num

    def forward(self, inputs, targets, images, depth_gt=None, interpolate=True):
        if interpolate:
            targets = F.interpolate(targets, inputs.shape[-2:], mode='bilinear', align_corners=True)
            images = F.interpolate(images, inputs.shape[-2:], mode='bilinear', align_corners=True)
            depth_gt = F.interpolate(depth_gt, inputs.shape[-2:], mode='bilinear', align_corners=True)
            
        n, _, _, _= inputs.size()
        
        # strict_mask is a range mask
        strict_mask = torch.logical_and(depth_gt>self.min_depth, depth_gt<self.max_depth)
        
        # remove pl out of range pixels
        invalid_pl_mask = targets == 80
        strict_mask = torch.logical_and(strict_mask, torch.logical_not(invalid_pl_mask))
        
        if self.only_missing_area:
            masks = depth_gt == self.missing_value # only consider missing values in semi loss
        else:
            masks = torch.ones_like(strict_mask).bool()

        edges_img, thetas_img = self.getEdge(images)
        
        # initialization
        loss = torch.DoubleTensor([0.0]).cuda()
        sample_num_sum = torch.tensor([0.0])
        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, sample_num_e = self.edgeGuidedSampling(
                inputs[i].squeeze(), 
                targets[i].squeeze(), 
                edges_img[i].squeeze(), 
                thetas_img[i].squeeze(), 
                masks[i].squeeze(), 
                depth_gt[i].squeeze(),
                strict_mask[i].squeeze())
            sample_num_sum += sample_num_e
            
            if sample_num_e == 0:
                continue
            if len(inputs_A) == 0 or len(inputs_B) == 0 or len(targets_A) == 0 or len(targets_B) == 0:
                continue
            
            # try:
            #     inputs_A_r, inputs_B_r, targets_A_r, targets_B_r, sample_num_r = self.randomSampling(
            #         inputs[i].squeeze().view(-1), 
            #         targets[i].squeeze().view(-1), 
            #         masks[i].squeeze().view(-1), 
            #         strict_mask[i].squeeze().view(-1), 
            #         sample_num_e)
            #     sample_num_sum += sample_num_r
            
            #     # Combine EGS + RS
            #     inputs_A = torch.cat((inputs_A, inputs_A_r), 0)
            #     inputs_B = torch.cat((inputs_B, inputs_B_r), 0)
            #     targets_A = torch.cat((targets_A, targets_A_r), 0)
            #     targets_B = torch.cat((targets_B, targets_B_r), 0)
                
            # except TypeError as e:
            #     print_log(e, logger='current')
                
                
            # GT ordinal relationship
            target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            target_weight = torch.abs(targets_A - targets_B) / (torch.max(torch.abs(targets_A - targets_B)) + 1e-6) # avoid nan
            target_weight = torch.exp(target_weight)
            # target_weight = torch.abs(targets_A - targets_B)
            # target_weight = torch.exp(target_weight)
            
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            if self.reweight_target:
                equal_loss = (inputs_A - inputs_B).pow(2) * mask_eq.double() # can also use the weight
            else:
                equal_loss = (inputs_A - inputs_B).pow(2) / target_weight * mask_eq.double() # can also use the weight
                
            
            if self.reweight_target:
                unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) / target_weight * labels)) * (~mask_eq).double()
            else:
                unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels)) * (~mask_eq).double()

            # Please comment the regularization term if you don't want to use the multi-scale gradient matching loss !!!
            loss = loss + self.alpha * equal_loss.mean() + 1.0 * unequal_loss.mean() #+ 0.2 * regularization_loss.double()
        
        self.idx += 1
        return loss[0].float()/n, float(sample_num_sum/n)
    

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

@MODELS.register_module()
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        _, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = target.shape
    
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        return loss

@MODELS.register_module()
class ExistLoss(nn.Module):
    """ExistLoss loss (pixel-wise)"""
    def __init__(self, reweight_target):
        super(ExistLoss, self).__init__()
        self.name = 'ExistLoss'
        self.reweight_target = reweight_target


    def forward(self, pred_grad, pl_grad, pseudo_edge_area):
        
        pred_grad_edge = pred_grad[pseudo_edge_area]
        pl_grad_edge = pl_grad[pseudo_edge_area]
        pl_grad_weight = torch.exp(pl_grad_edge)
        
        if self.reweight_target:
            loss = torch.exp(-pred_grad_edge / pl_grad_weight).mean()
        else:
            loss = torch.exp(-pred_grad_edge).mean()
        return loss
