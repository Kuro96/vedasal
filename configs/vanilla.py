# 1. data
dataset_type = 'AdFramesDataset'
data_root = 'data/advertisement/pc_all/'
# data_root = 'data/advertisement/pc_batch2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        typename=dataset_type,
        videos_path=data_root + 'train/images/',
        maps_path=data_root + 'train/annotations/',
        pipeline=[
            dict(typename='SampleFrames', clip_length=12),
            dict(typename='RawFrameDecode'),
            dict(typename='RawMapDecode'),
            dict(typename='Resize', scale=(384, 224), keep_ratio=True),
            dict(typename='RandomFlip', flip_ratio=0.0),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size_divisor=32),
            dict(typename='ToFloatTensor', keys=['frames', 'maps']),
            dict(typename='FormatShape', in_format='THWC', out_format='TCHW'),
            dict(typename='Collect', keys=['frames', 'maps'])]
    ),
    val=dict(
        typename=dataset_type,
        videos_path=data_root + 'val/images/',
        maps_path=data_root + 'val/annotations/',
        pipeline=[
            dict(typename='SampleFrames'),
            dict(typename='RawFrameDecode'),
            dict(typename='RawMapDecode'),
            dict(typename='Resize', scale=(384, 224), keep_ratio=True),
            dict(typename='RandomFlip', flip_ratio=0.0),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size_divisor=32),
            dict(typename='ToFloatTensor', keys=['frames', 'maps']),
            dict(typename='FormatShape', in_format='THWC', out_format='TCHW'),
            dict(typename='Collect', keys=['frames', 'maps'])]
    ),
    test=dict(
        typename=dataset_type,
        videos_path=data_root + 'val/images/',
        maps_path=data_root + 'val/annotations/',
        pipeline=[
            dict(typename='SampleFrames'),
            dict(typename='RawFrameDecode'),
            dict(typename='RawMapDecode'),
            dict(typename='Resize', scale=(384, 224), keep_ratio=True),
            dict(typename='RandomFlip', flip_ratio=0.0),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size_divisor=32),
            dict(typename='ToFloatTensor', keys=['frames', 'maps']),
            dict(typename='FormatShape', in_format='THWC', out_format='TCHW'),
            dict(typename='Collect', keys=['frames', 'maps'])]
    )
)

# 2. model
model = dict(
    typename='VanillaSal',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # TODO
        norm_cfg=dict(typename='BN', requires_grad=True),  # TODO
        norm_eval=False,
        style='pytorch',  # TODO
    ),
    neck=dict(
        typename='FPNFusion',
        # in_channels=[64, 128, 256, 512],
        in_channels=[256, 512, 1024, 2048],
        out_channels=256
    ),
    sequencer=dict(
        typename='ConvLSTMCell',
        in_channels=256,
        hidden_channels=256,
        with_cell_state=False
    ),
    head=dict(
        typename='VanillaHead',
        in_channels=256
    ))

# 3. engine
modes = ('train',)  # 'val')
modes = ('val',)  # 'val')
train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='PointwiseCriterion',
        loss=dict(typename='KLDLoss', reduction='batchmean')),
    optimizer=dict(
        typename='Adam',
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4),
)
val_engine = dict(typename='ValEngine', model=model, split_size=12)
infer_engine = dict(typename='InferEngine', model=model)

# 4. hooks
hooks = [
    dict(typename='OptimizerHook'),
    dict(
        typename='StepLrSchedulerHook',
        step=[53, 73],
        warmup='linear',
        warmup_iters=10,
        warmup_ratio=0.01),
    dict(typename='EvalHook'),
    dict(typename='SnapshotHook', interval=1),
    dict(typename='LoggerHook', interval=1)  # interval=dict(train=10, val=20))
]

# 5. runtime
max_epochs = 80
log_level = 'DEBUG'
