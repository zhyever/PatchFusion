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

# This file is partly inspired from ZoeDepth (https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/trainers/zoedepth_trainer.py); author: Shariq Farooq Bhat

import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss_sample import SILogLoss, DistributionLoss
from zoedepth.trainers.loss import SILogLoss as DenseSILogLoss
from zoedepth.trainers.loss import BudgetConstraint, HistogramMatchingLoss, SSIM, ConsistencyLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer, is_rank_zero, colors, flatten
from torchvision import transforms
from PIL import Image
import numpy as np

import wandb
import uuid
from tqdm import tqdm
from datetime import datetime as dt
import torch.distributed as dist

import copy
from zoedepth.utils.misc import generatemask
import torch.optim as optim

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        self.addf = config.get("addf", False)
        self.lazy_epoch = -1
        self.boostingdepth = config.get("boostingdepth", False)
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss(beta=config.get("beta", 0.15))
        self.dense_silog_loss = DenseSILogLoss(beta=config.get("beta", 0.15))
        print("sigloss's beta is set to {}".format(config.get("beta", 0.15)))
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.distribution_loss = DistributionLoss(max_depth=self.config.max_depth)
        self.sampled_training = config.get("sampled_training", False)
        self.sec_stage = config.get("sec_stage", False)
        self.multi_consistency = config.get("multi_consistency", False)
        self.use_blur = config.get("use_blur", False)
        self.dynamic = config.get("dynamic", False)
        if self.dynamic:
            self.dynamic_unupdate_rate = config.get("dynamic_unupdate_rate", 0.0)
            self.budget_loss = BudgetConstraint(loss_mu=0.0, flops_all=21552.5684, warm_up=True)
        self.use_scale_loss = config.get("use_scale_loss", False)
        if self.use_scale_loss:
            if config.get("scale_type", "ssim"):
                self.scale_loss = SSIM(window_size=config.get("window_size", int(11)))
            else:
                self.scale_loss = HistogramMatchingLoss(min_depth=self.config.min_depth, max_depth=self.config.max_depth)
        self.scale_target = config.get("scale_target", None)
        self.consistency_training = config.get("consistency_training", False)
        if self.consistency_training:
            self.consistency_target = config.get("consistency_target", None)
            self.consistency_loss = ConsistencyLoss(self.consistency_target, config.get("focus_flatten", False), config.get("w_p", 1.0))
            print("current weight for consistency loss is {}. focus_flatten is {}. w_p is {}".format(self.config.w_consistency, config.get("focus_flatten", False), config.get("w_p", 1.0)))
        
        

    def train_on_batch(self, batch, train_step, step_rate):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt = batch['image'].to(self.device), batch['depth'].to(self.device)
        image_raw = batch.get("image_raw", None)
        if image_raw is not None:
            image_raw = image_raw.to(self.device)
        sample_points = None
        if self.sampled_training:
            sample_points = batch['sample_points'].to(self.device)
        bbox = batch.get("bbox", None)
        if bbox is not None:
            bbox = bbox.to(self.device)
        bbox_raw = batch.get("bbox_raw", None)
        if bbox_raw is not None:
            bbox_raw = bbox_raw.to(self.device)
        depth_raw = batch.get("depth_raw", None)
        if depth_raw is not None:
            depth_raw = depth_raw.to(self.device)
        crop_area = batch.get("crop_area", None)
        if crop_area is not None:
            crop_area = crop_area.to(self.device)
        shift = batch.get("shift", None)
        if shift is not None:
            shift = shift.to(self.device)
        

        dataset = batch['dataset'][0]
        
        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)
        sample_mask = batch.get("sample_mask", None)
        if sample_mask is not None:
            sample_mask = sample_mask.to(self.device).to(torch.bool)
        mask_raw = batch.get("mask_raw", None)
        if mask_raw is not None:
            mask_raw = mask_raw.to(self.device).to(torch.bool)

        losses = {}

        with amp.autocast(enabled=self.config.use_amp):
            if self.sampled_training:
                output = self.model(images, sample_points, mode='train', image_raw=image_raw, bbox=bbox, depth_raw=depth_raw, crop_area=crop_area, shift=shift, bbox_raw=bbox_raw)
            else:
                output = self.model(images, None, mode='train', image_raw=image_raw, bbox=bbox, depth_raw=depth_raw, crop_area=crop_area, shift=shift, bbox_raw=bbox_raw)

            
            if self.boostingdepth:
                if self.lazy_epoch < self.epoch:
                    output.update_learning_rate()
                    self.lazy_epoch = self.epoch
                input_dict = dict()
                input_dict['data_gtfake'] = depths_gt
                output.set_input_train_gt(input_dict)
                output.optimize_parameters()
                pred_depths = output.fake_B
                pred = output.fake_B
                # print(torch.min(pred), torch.max(pred))
                losses = output.get_current_losses()

            else:
                pred_depths = output['metric_depth']

                if self.sampled_training:
                    sampled_depth_gt = sample_points[:, :, -1].float().unsqueeze(dim=-1)
                    sampled_depth_gt = sampled_depth_gt.permute(0, 2, 1)
                
                if self.config.get("representation", "") == 'biLaplacian':
                    # only for sampled training for now
                    l_dist, l_si = self.distribution_loss(output, sampled_depth_gt, mask=sample_mask)
                    loss = self.config.w_dist * l_dist + self.config.w_si * l_si
                    losses['distribution_loss'] = l_dist
                    losses['sigloss'] = l_si

                    if self.multi_consistency:
                        coarse, fine = output['coarse_depth_pred'], output['fine_depth_pred']
                        l_si_f = self.dense_silog_loss(
                            fine, depths_gt, mask=mask, interpolate=True, return_interpolated=False)
                        l_si_c = self.dense_silog_loss(
                            coarse, depth_raw, mask=mask_raw, interpolate=True, return_interpolated=False)
                        
                        losses['sigloss_f'] = l_si_f
                        losses['l_si_c'] = l_si_c
                        loss += self.config.w_si * (l_si_f + l_si_c)

                else:
                    if self.sampled_training:
                        l_si = self.silog_loss(
                            pred_depths, sampled_depth_gt, mask=sample_mask)
                        loss = self.config.w_si * l_si
                        losses[self.silog_loss.name] = l_si

                        if self.multi_consistency:
                            coarse, fine = output['coarse_depth_pred'], output['fine_depth_pred']
                            l_si_f = self.dense_silog_loss(
                                fine, depths_gt, mask=mask, interpolate=True, return_interpolated=False)
                            l_si_c = self.dense_silog_loss(
                                coarse, depth_raw, mask=mask_raw, interpolate=True, return_interpolated=False)
                            
                            losses['sigloss_f'] = l_si_f
                            losses['l_si_c'] = l_si_c
                            loss += self.config.w_si * (l_si_f + l_si_c)

                    else:
                        if self.multi_consistency:
                            #### here here here
                            pred_depths, coarse, fine = output['metric_depth'], output['coarse_depth_pred'], output['fine_depth_pred']

                            if self.consistency_training:
                                depths_gt = torch.split(depths_gt, 1, dim=1)
                                depths_gt = torch.cat(depths_gt, dim=0)
                                mask = torch.split(mask, 1, dim=-1)
                                mask = torch.cat(mask, dim=0).permute(0, 3, 1, 2)
                                mask_raw = torch.cat([mask_raw, mask_raw], dim=0)
                                depth_raw = torch.cat([depth_raw, depth_raw], dim=0)
                                temp_features = output.get('temp_features', None)
                                

                            l_si_1, pred = self.dense_silog_loss(
                                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
                            l_si_f, pred_f = self.dense_silog_loss(
                                fine, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
                            l_si_c = self.dense_silog_loss(
                                coarse, depth_raw, mask=mask_raw, interpolate=True, return_interpolated=False)
                            
                            losses[self.silog_loss.name] = l_si_1
                            losses['sigloss_f'] = l_si_f
                            losses['l_si_c'] = l_si_c
                            # loss = l_si_1 + l_si_f + l_si_c
                            loss = l_si_1

                            if self.consistency_training:
                                try:
                                    # depths_gt?  pred_f?
                                    l_consistency = self.consistency_loss(pred, shift, mask, temp_features, pred_f=depths_gt) # use the resized pred
                                except RuntimeError as e:
                                    print(e)
                                    print("some runtime error here! Hack with 0")
                                    l_consistency = torch.Tensor([0]).squeeze()

                                losses[self.consistency_loss.name] = l_consistency
                                loss += l_consistency * self.config.w_consistency

                        else:

                            l_si, pred = self.dense_silog_loss(
                                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)

                            loss = self.config.w_si * l_si
                            losses[self.silog_loss.name] = l_si

                if self.dynamic:
                    if step_rate > self.dynamic_unupdate_rate:
                        warm_up_rate = min(1.0, (step_rate - self.dynamic_unupdate_rate) / 0.02)
                        flop_cost = self.budget_loss(output['all_cell_flops'], warm_up_rate=warm_up_rate)
                        loss += self.config.w_flop * flop_cost
                        losses['flop_loss'] = flop_cost
                    else:
                        flop_cost = self.budget_loss(output['all_cell_flops'], warm_up_rate=1)
                        loss += 0 * flop_cost
                        losses['flop_loss'] = flop_cost

                if self.use_scale_loss:
                    if self.scale_target == 'coarse':
                        h_loss = self.scale_loss(pred_depths, output['coarse_depth_pred_roi'], mask, interpolate=True)
                    else:
                        h_loss = self.scale_loss(pred_depths, depths_gt, mask, interpolate=True)
                    loss += self.config.w_scale * h_loss
                    losses['scale_loss'] = h_loss

            
                # self.scaler.scale(loss).backward()

                # if self.config.clip_grad > 0:
                #     self.scaler.unscale_(self.optimizer)
                #     nn.utils.clip_grad_norm_(
                #         self.model.parameters(), self.config.clip_grad)

                # self.scaler.step(self.optimizer)
                
                # self.scaler.update()
                # self.optimizer.zero_grad()


                self.scaler.scale(loss).backward()

                if self.config.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad)

                self.scaler.step(self.optimizer)
                
                self.scaler.update()
                self.optimizer.zero_grad()


        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            if self.config.get("debug", False):
                pred = nn.functional.interpolate(
                    pred[0:1], depths_gt.shape[-2:], mode='bilinear', align_corners=True)[0]
                import matplotlib.pyplot as plt
                plt.imshow(pred.squeeze().detach().cpu().numpy())
                plt.savefig('debug.png')
                pass
            else:
                pred = nn.functional.interpolate(
                    pred[0:1], depths_gt.shape[-2:], mode='bilinear', align_corners=True)[0]
                depths_gt[torch.logical_not(mask)] = DATASETS_CONFIG[dataset]['max_depth']
                if self.consistency_training:
                    split_images = torch.split(images, 3, dim=1)
                    images = torch.cat(split_images, dim=0)
                self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred}, prefix="Train",
                                min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])
                
        return losses
    
    @torch.no_grad()
    def eval_infer(self, x, image_raw, bboxs=None, crop_area=None, dataset='u4k', bbox_raw=None):
        m = self.model.module if self.config.multigpu else self.model

        if dataset == 'u4k':
            base_h = 540
            base_w = 960
        elif dataset == 'gta':
            base_h = 270
            base_w = 480
        elif dataset == 'nyu':
            base_h = 120 * 2
            base_w = 160 * 2
        else:
            raise NotImplementedError
        
        if dataset == 'nyu':
            if self.sec_stage:
                images_crops = torch.split(x, 3, dim=1)
                bboxs_list = torch.split(bboxs, 1, dim=1)
                crop_areas = torch.split(crop_area, 1, dim=1)

                pred_depth_crops = []
                for i, (img, bbox, crop_area) in enumerate(zip(images_crops, bboxs_list, crop_areas)):
                    with amp.autocast(enabled=self.config.use_amp):
                        if i == 0:
                            out_dict = m(img, mode='eval', image_raw=image_raw, bbox=bbox[0], crop_area=crop_area, bbox_raw=bbox_raw[:, i, :] if bbox_raw is not None else None)
                            # whole_depth_pred = out_dict['coarse_depth_pred']
                            pred_depth_crop = out_dict['metric_depth']
                        else:
                            pred_depth_crop = m(img, mode='eval', image_raw=image_raw, bbox=bbox[0], crop_area=crop_area, bbox_raw=bbox_raw[:, i, :] if bbox_raw is not None else None)['metric_depth']
                            
                        pred_depth_crop = nn.functional.interpolate(
                            pred_depth_crop, (base_h, base_w), mode='bilinear', align_corners=True)
                        pred_depth_crops.append(pred_depth_crop)

                x_start, y_start = [0, base_h], [0, base_w]
                pred_depth = torch.zeros((base_h*2, base_w*2)).cuda()
                inner_idx = 0
                for ii, x in enumerate(x_start):
                    for jj, y in enumerate(y_start):
                        if self.use_blur:
                            pred_depth[x: x+base_h, y: y+base_w] = pred_depth_crops[inner_idx].squeeze() # do not care about boundry during validation
                        else:
                            pred_depth[x: x+base_h, y: y+base_w] = pred_depth_crops[inner_idx].squeeze()
                        inner_idx += 1
                pred_depth = pred_depth.squeeze(dim=0)

            else:

                with amp.autocast(enabled=self.config.use_amp):
                    pred_depth = m(x, mode='eval', image_raw=image_raw)['metric_depth']
                
        else:
            if self.sec_stage:
                images_crops = torch.split(x, 3, dim=1)
                bboxs_list = torch.split(bboxs, 1, dim=1)
                crop_areas = torch.split(crop_area, 1, dim=1)

                pred_depth_crops = []
                for i, (img, bbox, crop_area) in enumerate(zip(images_crops, bboxs_list, crop_areas)):
                    with amp.autocast(enabled=self.config.use_amp):
                        if i == 0:
                            out_dict = m(img, mode='eval', image_raw=image_raw, bbox=bbox[0], crop_area=crop_area, bbox_raw=bbox_raw[:, i, :] if bbox_raw is not None else None)
                            # whole_depth_pred = out_dict['coarse_depth_pred']
                            pred_depth_crop = out_dict['metric_depth']
                        else:
                            pred_depth_crop = m(img, mode='eval', image_raw=image_raw, bbox=bbox[0], crop_area=crop_area, bbox_raw=bbox_raw[:, i, :] if bbox_raw is not None else None)['metric_depth']
                            
                        pred_depth_crop = nn.functional.interpolate(
                            pred_depth_crop, (base_h, base_w), mode='bilinear', align_corners=True)
                        pred_depth_crops.append(pred_depth_crop)

                x_start, y_start = [0, base_h], [0, base_w]
                pred_depth = torch.zeros((base_h*2, base_w*2)).cuda()
                inner_idx = 0
                for ii, x in enumerate(x_start):
                    for jj, y in enumerate(y_start):
                        if self.use_blur:
                            pred_depth[x: x+base_h, y: y+base_w] = pred_depth_crops[inner_idx].squeeze() # do not care about boundry during validation
                        else:
                            pred_depth[x: x+base_h, y: y+base_w] = pred_depth_crops[inner_idx].squeeze()
                        inner_idx += 1
                pred_depth = pred_depth.squeeze(dim=0)

            else:

                with amp.autocast(enabled=self.config.use_amp):
                    pred_depth = m(x, mode='eval', image_raw=image_raw)['metric_depth']
        
        return pred_depth

    @torch.no_grad()
    def crop_aware_infer(self, x, image_raw):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths

    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        image_raw = batch['image_raw'].to(self.device)
        mask = batch["mask"].to(self.device)
        disp_gt_edges = batch['disp_gt_edges'].squeeze().numpy()
        bboxs = batch.get("bbox", None)
        if bboxs is not None:
            bboxs = bboxs.to(self.device)
        bbox_raw = batch.get("bbox_raw", None)
        if bbox_raw is not None:
            bbox_raw = bbox_raw.to(self.device)
        crop_area = batch.get("crop_area", None)
        if crop_area is not None:
            crop_area = crop_area.to(self.device)

        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
        # if dataset == 'nyu':
        #     pred_depths = self.crop_aware_infer(images, image_raw)
        # else:
        #     pred_depths = self.eval_infer(images, image_raw, bboxs, crop_area, dataset, bbox_raw)
        pred_depths = self.eval_infer(images, image_raw, bboxs, crop_area, dataset, bbox_raw)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
        # print(pred_depths.shape) # torch.Size([1, 1, 2160, 3840])
        # print(depths_gt.shape) # torch.Size([1, 1, 2160, 3840])

        with amp.autocast(enabled=self.config.use_amp):
            if self.sampled_training:
                l_depth = self.silog_loss(
                    pred_depths, depths_gt, mask=mask.to(torch.bool))
            else:
                l_depth = self.dense_silog_loss(
                    pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)



        metrics = compute_metrics(depths_gt, pred_depths, disp_gt_edges=disp_gt_edges, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if self.should_log and self.config.get("debug", False):
            print(metrics)
        if val_step in [21, 27] and self.should_log:
            if self.config.get("debug", False):
                pass
            else:
                if self.sec_stage:
                    log_rgb = image_raw
                else:
                    log_rgb = images

                scale_pred = nn.functional.interpolate(
                    pred_depths[0:1], depths_gt.shape[-2:], mode='bilinear', align_corners=True)[0]
                depths_gt[torch.logical_not(mask)] = DATASETS_CONFIG[dataset]['max_depth']
                self.log_images(rgb={"Input": log_rgb[0]}, depth={"GT": depths_gt[0], "PredictedMono": scale_pred}, prefix="Test",
                                min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses


    
    def train(self):
        print(f"Training {self.config.name}")
        if self.config.uid is None:
            self.config.uid = str(uuid.uuid4()).split('-')[-1]
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{self.config.uid}"
        self.config.run_id = run_id
        self.config.experiment_id = f"{self.config.wandb_start}_{self.config.name}{self.config.version_name}_{run_id}"
        self.should_write = ((not self.config.distributed)
                             or self.config.rank == 0)
        self.should_log = self.should_write  # and logging
        if self.should_log:
            if self.config.get("debug", False):
                pass
            else:
                tags = self.config.tags.split(
                    ',') if self.config.tags != '' else None
                wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root,
                        tags=tags, notes=self.config.notes, settings=wandb.Settings(start_method="fork"))

        self.model.train()
        self.step = 0
        best_loss = np.inf
        validate_every = int(self.config.validate_every * self.iters_per_epoch)


        if self.config.prefetch:

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"Prefetching...",
                                 total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader):
                pass
        
        losses = {}
        def stringify_losses(L): return "; ".join(map(
            lambda kv: f"{colors.fg.purple}{kv[0]}{colors.reset}: {round(kv[1].item(),3):.4e}", L.items()))
        
        epoch_len = len(self.train_loader)
        total_step = epoch_len * self.config.epochs

        for epoch in range(self.config.epochs):
            if self.should_early_stop():
                break
            
            self.epoch = epoch
            # self.save_checkpoint(f"{self.config.experiment_id}_latest.pt") # debug
            ################################# Train loop ##########################################################
            if self.should_log:
                if self.config.get("debug", False):
                    pass
                else:
                    wandb.log({"Epoch": epoch}, step=self.step)
            pbar = tqdm(enumerate(self.train_loader), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                        total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader)

            # 1532146.125
            for i, batch in pbar:
                current_step = epoch_len * epoch + i
                step_rate = current_step / total_step
                # metrics, test_losses = self.validate()
                # print(metrics)
                if self.should_early_stop():
                    print("Early stopping")
                    break
                # print(f"Batch {self.step+1} on rank {self.config.rank}")
                losses = self.train_on_batch(batch, i, step_rate)
                # print(f"trained batch {self.step+1} on rank {self.config.rank}")
                if self.config.get("debug", False):
                    log_info = ""
                    for name, loss in losses.items():
                        log_info += "{}: {}, ".format(name, loss)
                    print(log_info)

                if self.boostingdepth:
                    for k,v in losses.items():
                        losses[k] = torch.tensor(v)

                self.raise_if_nan(losses)
                if is_rank_zero(self.config) and self.config.print_losses:
                    pbar.set_description(
                        f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train. Losses: {stringify_losses(losses)}")
                self.scheduler.step()

                if self.should_log and self.step % 50 == 0:
                    if self.config.get("debug", False):
                        log_info = ""
                        for name, loss in losses.items():
                            log_info += "{}: {}, ".format(name, loss)
                        print(log_info)
                    else:
                        wandb.log({f"Train/{name}": loss.item()
                                for name, loss in losses.items()}, step=self.step)
                        # current_lr = self.optimizer.param_groups[0]['lr']
                        current_lr = self.scheduler.get_last_lr()[0]
                        wandb.log({f"Train/LR": current_lr}, step=self.step)
                        momentum = self.optimizer.param_groups[0]['betas'][0]
                        wandb.log({f"Train/momentum": momentum}, step=self.step)
                        wandb.log({f"Train/step_rate": step_rate}, step=self.step)

                self.step += 1

                ########################################################################################################

                if self.test_loader:
                    if (self.step % validate_every) == 0:
                        self.model.eval()
                        if self.should_write:
                            self.save_checkpoint(
                                f"{self.config.experiment_id}_latest.pt")

                        ################################# Validation loop ##################################################
                        # validate on the entire validation set in every process but save only from rank 0, I know, inefficient, but avoids divergence of processes
                        metrics, test_losses = self.validate()
                        # print("Validated: {}".format(metrics))
                        if self.should_log:
                            if self.config.get("debug", False):
                                log_info = ""
                                for name, loss in test_losses.items():
                                    log_info += "{}: {}, ".format(name, loss)
                                log_info = "\n"
                                for name, val in metrics.items():
                                    log_info += "{}: {}, ".format(name, val)
                                print(log_info)
                                
                            else:
                                wandb.log(
                                    {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)

                                wandb.log({f"Metrics/{k}": v for k,
                                        v in metrics.items()}, step=self.step)

                            if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                                self.save_checkpoint(
                                    f"{self.config.experiment_id}_best.pt")
                                best_loss = metrics[self.metric_criterion]

                        self.model.train()

                        if self.config.distributed:
                            dist.barrier()
                        # print(f"Validated: {metrics} on device {self.config.rank}")

                # print(f"Finished step {self.step} on device {self.config.rank}")
                #################################################################################################

        # Save / validate at the end
        self.step += 1  # log as final point
        self.model.eval()
        self.save_checkpoint(f"{self.config.experiment_id}_latest.pt")
        if self.test_loader:

            ################################# Validation loop ##################################################
            metrics, test_losses = self.validate()
            # print("Validated: {}".format(metrics))
            if self.should_log:
                if self.config.get("debug", False):
                    log_info = ""
                    for name, loss in test_losses.items():
                        log_info += "{}: {}, ".format(name, loss)
                    log_info = "\n"
                    for name, val in metrics.items():
                        log_info += "{}: {}, ".format(name, val)
                    print(log_info)
                    
                else:
                    wandb.log({f"Test/{name}": tloss for name,
                            tloss in test_losses.items()}, step=self.step)
                    wandb.log({f"Metrics/{k}": v for k,
                            v in metrics.items()}, step=self.step)

                if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                    self.save_checkpoint(
                        f"{self.config.experiment_id}_best.pt")
                    best_loss = metrics[self.metric_criterion]

        self.model.train()


    def init_optimizer(self):
        m = self.model.module if self.config.multigpu else self.model

        if self.config.same_lr:
            print("Using same LR")
            if hasattr(m, 'core'):
                m.core.unfreeze()
            params = self.model.parameters()
        else:
            print("Using diff LR")
            if not hasattr(m, 'get_lr_params'):
                raise NotImplementedError(
                    f"Model {m.__class__.__name__} does not implement get_lr_params. Please implement it or use the same LR for all parameters.")

            params = m.get_lr_params(self.config.lr)

        # if self.addf:
        #     return optim.Adam(params, lr=self.config.lr, betas=(0.5, 0.999))
        # else:
        #     return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)

        return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)


    
    def save_checkpoint(self, filename):
        if not self.should_write:
            return
        root = self.config.save_dir
        if not os.path.isdir(root):
            os.makedirs(root)

        fpath = os.path.join(root, filename)
        m = self.model.module if self.config.multigpu else self.model
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": None,  # TODO : Change to self.optimizer.state_dict() if resume support is needed, currently None to reduce file size
                "epoch": self.epoch
            }, fpath)
        
        if self.boostingdepth:
            fpath = os.path.join(root, "_fusion" + filename)
            m.fusion_network.save_networks(fpath)
