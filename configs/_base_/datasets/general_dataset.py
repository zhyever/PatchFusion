

general_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='ImageDataset',
        rgb_image_dir='',
        dataset_name='',
        gt_dir=None,
        network_process_size=(384, 512),
        resize_mode='zoe'))

