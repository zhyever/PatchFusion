_base_ = [
    '../_base_/datasets/general_dataset.py',
]

collect_input_args=['image_lr', 'depth_gt', 'image_hr']

general_dataloader=dict(
    dataset=dict(
        network_process_size=(392, 518),
        resize_mode='depth-anything'))