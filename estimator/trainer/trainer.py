import os
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.distributed as dist
from mmengine.dist import get_dist_info, collect_results_cpu, collect_results_gpu
from mmengine import print_log
import torch.nn.functional as F
from tqdm import tqdm
from estimator.utils import colorize

class Trainer:
    """
    Trainer class
    """
    def __init__(
        self, 
        config,
        runner_info,
        train_sampler,
        train_dataloader,
        val_dataloader,
        model):
       
        self.config = config
        self.runner_info = runner_info
        
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

        # build opt and schedule
        self.optimizer_wrapper = build_optim_wrapper(self.model, config.optim_wrapper)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer_wrapper.optimizer, [l['lr'] for l in self.optimizer_wrapper.optimizer.param_groups], epochs=self.config.train_cfg.max_epochs, steps_per_epoch=len(self.train_dataloader),
            cycle_momentum=config.param_scheduler.cycle_momentum, base_momentum=config.param_scheduler.get('base_momentum', 0.85), max_momentum=config.param_scheduler.get('max_momentum', 0.95),
            div_factor=config.param_scheduler.div_factor, final_div_factor=config.param_scheduler.final_div_factor, pct_start=config.param_scheduler.pct_start, three_phase=config.param_scheduler.three_phase)
    
        # I'd like use wandb log_name
        self.train_step = 0 # for training
        self.val_step = 0 # for validation

        self.iters_per_train_epoch = len(self.train_dataloader)
        self.iters_per_val_epoch = len(self.val_dataloader)
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.collect_input_args = config.collect_input_args
        print_log('successfully init trainer', logger='current')
    
    
    def log_images(self, log_dict, prefix="", scalar_cmap="turbo_r", min_depth=1e-3, max_depth=80, step=0):
        # Custom log images. Please add more items to the log dict returned from the model
        
        wimages = dict()
        wimages['{}/step'.format(prefix)] = step
        
        rgb = log_dict.get('rgb')[0]
        _, h_rgb, w_rgb = rgb.shape
        
        if 'depth_pred' in log_dict.keys():
            depth_pred = log_dict.get('depth_pred')[0]
            depth_pred = depth_pred.squeeze()
            depth_gt = log_dict.get('depth_gt')[0]
            depth_gt = depth_gt.squeeze()
            invalid_mask = torch.logical_or(depth_gt<=min_depth, depth_gt>=max_depth).detach().cpu().squeeze().numpy() # (h, w)
            
            if np.sum(np.logical_not(invalid_mask)) == 0: # all pixels in gt are invalid
                return
            
            depth_gt_color = colorize(depth_gt, vmin=None, vmax=None, invalid_mask=invalid_mask, cmap=scalar_cmap)
            depth_pred_color = colorize(depth_pred, vmin=None, vmax=None)
            
            depth_gt_img = wandb.Image(depth_gt_color, caption='depth_gt')
            depth_pred_img = wandb.Image(depth_pred_color, caption='depth_pred')
            rgb = wandb.Image(rgb, caption='rgb')
            
            wimages['{}/LogImageDepth'.format(prefix)] = [rgb, depth_gt_img, depth_pred_img]
            
        if 'seg_pred' in log_dict.keys():
            seg_pred = log_dict.get('seg_pred')[0]
            seg_pred = seg_pred.squeeze()
            seg_gt = log_dict.get('seg_gt')[0]
            seg_gt = seg_gt.squeeze()
            # class_labels = {0: "good", 1: "refine", 2: "oor", 3: "sky"}
            class_labels = {0: "bg", 1: "edge"}
            
            mask_img = wandb.Image(
                rgb,
                masks={
                    "predictions": {"mask_data": seg_pred.detach().cpu().numpy(), "class_labels": class_labels},
                    "ground_truth": {"mask_data": seg_gt.detach().cpu().numpy(), "class_labels": class_labels},
                },
                caption='segmentation')
            wimages['{}/LogImageSeg'.format(prefix)] = [mask_img]
        
        if 'mask' in log_dict.keys():
            mask = log_dict.get('mask')[0]
            mask = mask.squeeze().float()*255
            mask_img = wandb.Image(
                mask.detach().cpu().numpy(),
                caption='segmentation')
            cur_log = wimages['{}/LogImageDepth'.format(prefix)]
            cur_log.append(mask_img)
            wimages['{}/LogImageDepth'.format(prefix)] = cur_log
        
        # some other things
        if 'pseudo_gt' in log_dict.keys():
            pseudo_gt = log_dict.get('pseudo_gt')[0]
            pseudo_gt = pseudo_gt.squeeze()
            pseudo_gt_color = colorize(pseudo_gt, vmin=None, vmax=None, cmap=scalar_cmap)
            pseudo_gt_img = wandb.Image(pseudo_gt_color, caption='pseudo_gt')
            cur_log = wimages['{}/LogImageDepth'.format(prefix)]
            cur_log.append(pseudo_gt_img)
            # pseudo_gt = log_dict.get('pseudo_gt')[0][0]
            # pseudo_gt = pseudo_gt * 255
            # pseudo_gt = pseudo_gt.astype(np.uint8)
            # pseudo_gt_img = wandb.Image(pseudo_gt, caption='pseudo_gt')
            # cur_log = wimages['{}/LogImageDepth'.format(prefix)]
            # cur_log.append(pseudo_gt_img)
            
        wandb.log(wimages)
            

    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
                    
                    
    @torch.no_grad()
    def val_epoch(self):
        results = []
        results_list = [[] for _ in range(8)]
        
        self.model.eval()
        dataset = self.val_dataloader.dataset
        loader_indices = self.val_dataloader.batch_sampler

        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.val_dataloader)):
            self.val_step += 1

            batch_data_collect = self.collect_input(batch_data)
            # result, log_dict = self.model(mode='infer',  **batch_data_collect)
            result, log_dict = self.model(mode='infer', cai_mode='m1', process_num=4, **batch_data_collect) # might use test/val to split cases

            if isinstance(result, list):
                # in case you have multiple results
                for num_res in range(len(result)):
                    metrics = dataset.get_metrics(
                        batch_data_collect['depth_gt'], 
                        result[num_res], 
                        disp_gt_edges=batch_data.get('boundary', None), 
                        additional_mask=log_dict.get('mask', None),
                        image_hr=batch_data.get('image_hr', None))
                    results_list[num_res].extend([metrics])
            
            else:
                metrics = dataset.get_metrics(
                    batch_data_collect['depth_gt'], 
                    result, 
                    seg_image=batch_data_collect.get('seg_image', None),
                    disp_gt_edges=batch_data.get('boundary', None), 
                    additional_mask=log_dict.get('mask', None), 
                    image_hr=batch_data.get('image_hr', None))
                results.extend([metrics])

            if self.runner_info.rank == 0:
                if isinstance(result, list):
                    batch_size = len(result[0]) * world_size
                else:
                    batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()

            if self.runner_info.rank == 0 and self.config.debug == False and (idx + 1) % self.config.train_cfg.val_log_img_interval == False:
                self.log_images(log_dict=log_dict, prefix="Val", min_depth=self.config.min_depth, max_depth=self.config.max_depth, step=self.val_step)
            
        # collect results from all ranks
        if isinstance(result, list):
            results_collect = []
            for results in results_list:
                results = collect_results_gpu(results, len(dataset))
                results_collect.append(results)
        else:
            results = collect_results_gpu(results, len(dataset))
            
        if self.runner_info.rank == 0:
            if isinstance(result, list):
                for num_refine in range(len(result)):
                    ret_dict = dataset.evaluate(results_collect[num_refine])
            else:
                ret_dict = dataset.evaluate(results)

        if self.runner_info.rank == 0 and self.config.debug == False:
            wdict = dict()
            for k, v in ret_dict.items():
                wdict["Val/{}".format(k)] = v.item()
            wdict['Val/step'] = self.val_step
            wandb.log(wdict)
        
        torch.cuda.empty_cache()
        if self.runner_info.distributed is True:
            torch.distributed.barrier()
        
        self.model.train() # avoid changing model state 
        
        
    def train_epoch(self, epoch_idx):
        
        self.model.train()
        if self.runner_info.distributed:
            dist.barrier()
        
        pbar = tqdm(enumerate(self.train_dataloader), desc=f"Epoch: {epoch_idx + 1}/{self.config.train_cfg.max_epochs}. Loop: Train",
                    total=self.iters_per_train_epoch) if self.runner_info.rank == 0 else enumerate(self.train_dataloader)
        for idx, batch_data in pbar:
            self.train_step += 1

            batch_data_collect = self.collect_input(batch_data)
            loss_dict, log_dict = self.model(mode='train', **batch_data_collect)
            
            total_loss = loss_dict['total_loss']
            # total_loss = self.grad_scaler.scale(loss_dict['total_loss'])
            
            self.optimizer_wrapper.update_params(total_loss)
            self.scheduler.step()
            
            # log something here
            if self.runner_info.rank == 0:
                log_info = 'Epoch: [{:02d}/{:02d}]'.format(epoch_idx + 1, self.config.train_cfg.max_epochs, idx + 1, len(self.train_dataloader))
                for k, v in loss_dict.items():
                    log_info += ' - {}: {:.2f}'.format(k, v.item())
                pbar.set_description(log_info)
                    
            if (idx + 1) % self.config.train_cfg.log_interval == 0:
                log_info = 'Epoch: [{:02d}/{:02d}] - Step: [{:05d}/{:05d}] - Time: [{}/{}] - Total Loss: {}'.format(epoch_idx + 1, self.config.train_cfg.max_epochs, idx + 1, len(self.train_dataloader), 1, 1, total_loss)
                for k, v in loss_dict.items():
                    if k != 'total_loss':
                        log_info += ' - {}: {}'.format(k, v)
                print_log(log_info, logger='current')
                
                if self.runner_info.rank == 0 and self.config.debug == False:
                    wdict = dict()
                    wdict['Train/total_loss'] = total_loss.item()
                    wdict['Train/LR'] = self.optimizer_wrapper.get_lr()['lr'][0]
                    wdict['Train/momentum'] = self.optimizer_wrapper.get_momentum()['momentum'][0]
                    wdict['Train/step'] = self.train_step
                    for k, v in loss_dict.items():
                        if k != 'total_loss':
                            if isinstance(v, torch.Tensor):
                                wdict['Train/{}'.format(k)] = v.item()
                            else:
                                wdict['Train/{}'.format(k)] = v
                    wandb.log(wdict)
   
            if self.runner_info.rank == 0 and self.config.debug == False and (idx + 1) % self.config.train_cfg.train_log_img_interval == False:
                self.log_images(log_dict=log_dict, prefix="Train", min_depth=self.config.min_depth, max_depth=self.config.max_depth, step=self.train_step)
            
            if self.config.train_cfg.val_type == 'iter_base':
                if (self.train_step + 1) % self.config.train_cfg.val_interval == 0 and (self.train_step + 1) >= self.config.train_cfg.get('eval_start', 0):
                    self.val_epoch()
                
    def save_checkpoint(self, epoch_idx):
        # As default, the model is wrappered by DDP!!! Hence, even if you're using one gpu, please use dist_train.sh
        if hasattr(self.model.module, 'get_save_dict'):
            print_log('Saving ckp, but use the inner get_save_dict fuction to get model_dict', logger='current')
            # print_log('For saving space. Would you like to save base model several times? :>', logger='current')
            model_dict = self.model.module.get_save_dict()
        else:
            model_dict = self.model.module.state_dict() 
            
        checkpoint_dict = {
            'epoch': epoch_idx, 
            'model_state_dict': model_dict, 
            'optim_state_dict': self.optimizer_wrapper.state_dict(),
            'schedule_state_dict': self.scheduler.state_dict()}
        
        if self.runner_info.rank == 0:
            torch.save(checkpoint_dict, os.path.join(self.runner_info.work_dir, 'checkpoint_{:02d}.pth'.format(epoch_idx + 1)))
        log_info = 'save checkpoint_{:02d}.pth at {}'.format(epoch_idx + 1, self.runner_info.work_dir)
        print_log(log_info, logger='current')
    
    def run(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print_log('training param: {}'.format(name), logger='current')
        
        # self.val_epoch() # do you want to debug val step?
        for epoch_idx in range(self.config.train_cfg.max_epochs):
            if self.runner_info.distributed:
                self.train_sampler.set_epoch(epoch_idx)
            self.train_epoch(epoch_idx) 
            if (epoch_idx + 1) % self.config.train_cfg.val_interval == 0 and (epoch_idx + 1) >= self.config.train_cfg.get('eval_start', 0) and self.config.train_cfg.val_type == 'epoch_base':
                self.val_epoch()
            if (epoch_idx + 1) % self.config.train_cfg.save_checkpoint_interval == 0:
                self.save_checkpoint(epoch_idx)
            if (epoch_idx + 1) % self.config.train_cfg.get('early_stop_epoch', 9999999) == 0: # Are you using 99999999+ epochs?
                print_log('early stop at epoch: {}'.format(epoch_idx), logger='current')
                break
        
        if self.config.train_cfg.val_type == 'iter_base':
            self.val_epoch()
            