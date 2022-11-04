_base_ = [
    "../_base_/models/mask_rcnn_swin_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(bbox_head=dict(num_classes=7), mask_head=dict(num_classes=7)),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480 * 1.5, 1332 * 1.5),
                        (512 * 1.5, 1332 * 1.5),
                        (544 * 1.5, 1332 * 1.5),
                        (576 * 1.5, 1332 * 1.5),
                        (608 * 1.5, 1332 * 1.5),
                        (640 * 1.5, 1332 * 1.5),
                        (672 * 1.5, 1332 * 1.5),
                        (704 * 1.5, 1332 * 1.5),
                        (736 * 1.5, 1332 * 1.5),
                        (768 * 1.5, 1332 * 1.5),
                        (800 * 1.5, 1332 * 1.5),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (400 * 1.5, 1332 * 1.5),
                        (500 * 1.5, 1332 * 1.5),
                        (600 * 1.5, 1332 * 1.5),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(576, 900),
                    allow_negative_crop=True,
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (480 * 1.5, 1332 * 1.5),
                        (512 * 1.5, 1332 * 1.5),
                        (544 * 1.5, 1332 * 1.5),
                        (576 * 1.5, 1332 * 1.5),
                        (608 * 1.5, 1332 * 1.5),
                        (640 * 1.5, 1332 * 1.5),
                        (672 * 1.5, 1332 * 1.5),
                        (704 * 1.5, 1332 * 1.5),
                        (736 * 1.5, 1332 * 1.5),
                        (768 * 1.5, 1332 * 1.5),
                        (800 * 1.5, 1332 * 1.5),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
dataset_type = "CocoDataset"
classes = (
    "ascopore",
    " autozooid",
    "avicularia",
    "opesia",
    "orifice",
    "ovicell",
    "spiramen",
)
data = dict(
    train=dict(
        type=dataset_type,
        img_prefix="../../images/train/",
        classes=classes,
        ann_file="../../images/val/annotations.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_prefix="../../images/val/",
        classes=classes,
        ann_file="../../images/val/annotations.json",
    ),
    test=dict(
        type=dataset_type,
        img_prefix="../../images/test/",
        classes=classes,
        ann_file="../../images/test/instances_default.json",
    ),
)
load_from = "../../inference/deepbryo.pth"


optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
lr_config = dict(step=[30, 60])
runner = dict(type="EpochBasedRunner", max_epochs=90)
