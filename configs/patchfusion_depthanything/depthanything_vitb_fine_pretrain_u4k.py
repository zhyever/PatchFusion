_base_ = [
    '../_base_/datasets/u4k.py', 
]

min_depth=1e-3
max_depth=80
    
zoe_depth_config=dict(
    type='DA-ZoeDepth',
    
    min_depth=min_depth,
    max_depth=max_depth,
    
    depth_anything=True,
    midas_model_type='vitb',
    img_size=[392, 518],
        
    # some important params
    # midas_model_type='DPT_BEiT_L_384',
    pretrained_resource='local::./work_dir/DepthAnything_vitb.pt',
    use_pretrained_midas=True,
    train_midas=True,
    freeze_midas_bn=True,
    do_resize=False, # do not resize image in midas

    # default settings
    attractor_alpha=1000,
    attractor_gamma=2,
    attractor_kind='mean',
    attractor_type='inv',
    aug=True,
    bin_centers_type='softplus',
    bin_embedding_dim=128,
    clip_grad=0.1,
    dataset='nyu',
    distributed=True,
    force_keep_ar=True,
    gpu='NULL',
    inverse_midas=False,
    log_images_every=0.1,
    max_temp=50.0,
    max_translation=100,
    memory_efficient=True,
    min_temp=0.0212,
    model='zoedepth',
    n_attractors=[16, 8, 4, 1],
    n_bins=64,
    name='ZoeDepth',
    notes='',
    output_distribution='logbinomial',
    prefetch=False,
    print_losses=False,
    project='ZoeDepth',
    random_crop=False,
    random_translate=False,
    root='.',
    save_dir='',
    shared_dict='NULL',
    tags='',
    translate_prob=0.2,
    uid='NULL',
    use_amp=False,
    use_shared_dict=False,
    validate_every=0.25,
    version_name='v1',
    workers=16,
)

model=dict(
    type='BaselinePretrain',
    patch_process_shape=(392, 518),
    min_depth=min_depth,
    max_depth=max_depth,
    target='fine',
    coarse_branch=zoe_depth_config,
    fine_branch=zoe_depth_config,
    sigloss=dict(type='SILogLoss'))

collect_input_args=['image_lr', 'crops_image_hr', 'depth_gt', 'crop_depths', 'bboxs', 'image_hr']

project='patchfusion'

train_cfg=dict(max_epochs=24, val_interval=2, save_checkpoint_interval=24, log_interval=100, train_log_img_interval=500, val_log_img_interval=50, val_type='epoch_base', eval_start=0)

optim_wrapper=dict(
    # optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.0002/50, weight_decay=0.01),
    clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
        }))

param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=1,
    final_div_factor=10000,
    pct_start=0.5,
    three_phase=False,)

env_cfg=dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='forkserver'),
    dist_cfg=dict(backend='nccl'))

convert_syncbn=True
find_unused_parameters=True

train_dataloader=dict(
    dataset=dict(
        resize_mode='depth-anything',
        transform_cfg=dict(
            network_process_size=[392, 518])))

val_dataloader=dict(
    dataset=dict(
        resize_mode='depth-anything',
        transform_cfg=dict(
            network_process_size=[392, 518])))