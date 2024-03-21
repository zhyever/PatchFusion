

train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='train',
        data_root='./data/u4k',
        split='./data/u4k/splits/train.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            degree=1.0,
            random_crop=True, # random_crop_size will be set as patch_raw_shape
            network_process_size=[384, 512])))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='infer',
        data_root='./data/u4k',
        split='./data/u4k/splits/val.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))

test_in_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='infer',
        data_root='./data/u4k',
        split='./data/u4k/splits/test.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))


test_out_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UnrealStereo4kDataset',
        mode='infer',
        data_root='./data/u4k',
        split='./data/u4k/splits/test_out.txt',
        min_depth=1e-3,
        max_depth=80,
        transform_cfg=dict(
            network_process_size=[384, 512])))