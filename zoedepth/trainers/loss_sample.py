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
import torch.cuda.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'


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

    def forward(self, input, target, mask=None):
        input = extract_key(input, KEY_OUTPUT)
        
        if mask is not None:
            input_filtered = input[mask]
            target_filtered = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input_filtered + alpha) - torch.log(target_filtered + alpha)
            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)
            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        return loss
    


import torch
import torch.nn.functional as F

def gaussian(mu, sigma, labels):
    return torch.exp(-0.5*(mu-labels)** 2/ sigma** 2)/sigma

def laplacian(mu, b, labels):
    # a = torch.abs(mu-labels)/b
    # print("1 isnan: {}".format(torch.isnan(a).any()))
    # print("1 isinf: {}".format(torch.isinf(a).any()))
    # a = torch.exp(-(torch.abs(mu-labels)/b))
    # print(a)

    # print("1 isnan: {}".format(torch.isnan(a).any()))
    # print("1 isinf: {}".format(torch.isinf(a).any()))
    return 0.5 * torch.exp(-(torch.abs(mu-labels)/b))/b

def distribution(mu, sigma, labels, dist="gaussian"):
    return gaussian(mu, sigma, labels) if dist=="gaussian" else \
           laplacian(mu, sigma, labels)

def bimodal_loss(mu0, mu1, sigma0, sigma1, w0, w1, labels, dist="gaussian"):
    # first_term = w0 * distribution(mu0, sigma0, labels, dist)
    # print(first_term)
    # print("f isnan: {}".format(torch.isnan(first_term).any()))
    # print("f isinf: {}".format(torch.isinf(first_term).any()))
    # second_term = w1 * distribution(mu1, sigma1, labels, dist)
    # print(second_term)
    # print("s isnan: {}".format(torch.isnan(second_term).any()))
    # print("s isinf: {}".format(torch.isinf(second_term).any()))

    loss = w0 * distribution(mu0, sigma0, labels, dist) + w1 * distribution(mu1, sigma1, labels, dist)
    # loss = torch.clamp(loss, min=1e-12)
    # print(loss)
    return - torch.log(loss)

def unimodal_loss(mu, sigma, labels):
    return torch.abs(mu - labels)/sigma + torch.log(sigma)

def smooth_l1_loss(preds, labels, reduce=None):
    return F.smooth_l1_loss(preds, labels, reduce=reduce)

def l1_loss(preds, labels, reduce=None):
    return F.l1_loss(preds, labels, reduce=reduce)


class DistributionLoss(nn.Module):
    def __init__(self, max_depth):
        super(DistributionLoss, self).__init__()
        self.name = 'DistributionLoss'
        self.max_depth = max_depth

    def forward(self, input, target, mask=None, dist='biLaplacian'):
        
        
        mu0 = input['mu0']
        mu1 = input['mu1']
        sigma0 = input['sigma0']
        sigma1 = input['sigma1']
        pi0 = input['pi0']
        pi1 = input['pi1']
        
        pred_mask = (pi0 / sigma0 > pi1 / sigma1).float()
        pred_depth = (mu0 * pred_mask + mu1 * (1. - pred_mask))
        pred_metric_depth = (1 - pred_depth) * self.max_depth


        if mask is not None:
            mu0 = mu0[mask]
            mu1 = mu1[mask]
            sigma0 = sigma0[mask]
            sigma1 = sigma1[mask]
            pi0 = pi0[mask]
            pi1 = pi1[mask]

            # real_input = real_depth[mask]
            
            real_input = mu0
            pred_metric_depth = pred_metric_depth[mask]
            record_target = target[mask]


        target_filtered = 1 - target[mask] / self.max_depth
        bi_loss = bimodal_loss(mu0, mu1, sigma0, sigma1, pi0, pi1, target_filtered, dist=dist).mean()
        # print(bi_loss)  

        alpha = 1e-7
        beta = 0.15
        g = torch.log(real_input + alpha) - torch.log(record_target + alpha)
        Dg = torch.var(g) + beta * torch.pow(torch.mean(g), 2)
        sig_loss = 10 * torch.sqrt(Dg)
        # print(sig_loss)
        
        return bi_loss, sig_loss


        
        