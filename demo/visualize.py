# -*- coding: utf-8 -*-
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copyright (c) 2022 E.V.A.S (E.Vehicle Autonomous Silicon) Intelligence
# 
# All rights are reserved. Reproduction in whole or in part is prohibited
# without the written consent of the copyright owner
# EVAS reserves the right to make changes without notice at any time.
# 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# @brief  visualize 
# @author coson.jia@evas.ai
# @date : 2023/3/16 上午11:17
# @file : visualize.py
#
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import argparse
import sys
import os
import torch
import numpy as np
import mmcv
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
import re
import cv2
import time


def get_cam_name(path):
    pattern = "/(CAM[A-Z_]*)/"
    match = re.search(pattern=pattern, string=path)
    if match is not None:
        return match.group(1)
    return None


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument("--mode", type=str, choices=['gt', 'pred'])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--bbox-score", type=float, default=.3)
    args = parser.parse_args()
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    # cfg.merge_from_dict(opts)
    return cfg, args


def main(cfg, args):
    # build the dataloader
    from mmdet3d.datasets import build_dataset, build_dataloader
    cfg_test = cfg.data.test
    if args.mode == 'pred':
        pipeline = cfg_test.pipeline
        for p in pipeline:
            if p.type == 'LoadAnnotations3D':
                break
        pipeline.remove(p)
    else:
        types = ['LoadImageFromFile','KittiSetOrigin', 'NormalizeMultiviewImage', ]
        pipeline = cfg_test.pipeline
        del_list = []
        for p in pipeline:
            if p.type in types:
                del_list.append(p)
        for p in del_list:
            pipeline.remove(p)

    dataset = build_dataset(cfg_test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    # build the model and load checkpoint
    if args.mode == 'pred':
        from mmdet3d.models import build_model
        model = build_model(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            from mmcv.runner import wrap_fp16_model
            wrap_fp16_model(model)
        from mmcv.runner import load_checkpoint
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            from mmcv.cnn import fuse_conv_bn
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE
        from mmcv.parallel import MMDataParallel
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
    from tqdm import tqdm
    start = time.time()
    for i, data in enumerate(tqdm(data_loader, file=sys.stdout)):
        print(f"\nloading data time: {time.time() - start}")
        start = time.time()
        metas = data['img_metas'].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])
        if args.mode == 'pred':
            with torch.inference_mode():
                outputs = model(return_loss=False, rescale=True, **data)
            bboxes = outputs[0]["boxes_3d"]  # .tensor.numpy()
            # bboxes[:, 7:7 + 2] = bboxes[:, 7:7 + 2] + metas['velocity'].reshape(1, 2)
            # bboxes = LiDARInstance3DBoxes(
            #     bboxes, box_dim=bboxes.shape[-1], origin=(0.5, 0.5, 0.0))
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()
            # if args.bbox_classes is not None:
            #     indices = np.isin(labels, args.bbox_classes)
            #     bboxes = bboxes[indices]
            #     scores = scores[indices]
            #     labels = labels[indices]
            #
            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            # bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9, origin=(0.5, 0.5, .5))
        elif args.mode == 'gt' and 'gt_bboxes_3d' in data:
            bboxes = data["gt_bboxes_3d"].data[0][0]  # .tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()
            # bboxes[..., 2] -= bboxes[..., 5] / 2
            # bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None
        print(f"inference time: {time.time() - start}")
        start = time.time()
        frame_cam = None
        if "img" in data:
            img_res = []
            for k, image_path in enumerate(metas["image_paths"]):
                cam_name = get_cam_name(image_path)
                if cam_name is None:
                    cam_name = f"camera-{k}"
                image = mmcv.imread(image_path)
                tmp_img = visualize_camera(
                    fpath=os.path.join(args.out_dir, f"score-{args.bbox_score}", cam_name, f"{name}.png"),
                    image=image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.class_names,
                    thickness=1,
                    title=cam_name
                )
                img_res.append(tmp_img)
            if len(img_res) == 6:
                front = np.concatenate([img_res[2], img_res[0], img_res[1]], axis=1)
                back = np.concatenate([img_res[4], img_res[3], img_res[5]], axis=1)
                frame_cam = np.concatenate([front, back], axis=0)
        print(f"draw camera time: {time.time() - start}")
        start = time.time()
        point_img = None
        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            point_img = visualize_lidar(
                os.path.join(args.out_dir, f"score-{args.bbox_score}", "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.class_names,
                save=True
            )
        print(f"draw point time: {time.time() - start}")
        start = time.time()
        if frame_cam is not None and point_img is not None:
            cam_h, cam_w, _ = frame_cam.shape
            p_h, p_w, _ = point_img.shape
            new_h = cam_h
            new_w = int(new_h * 1.0 / p_h * p_w)
            new_img = cv2.resize(point_img, (new_w, new_h))
            frame = np.concatenate([frame_cam, new_img], axis=1)
        elif frame_cam is not None:
            frame = frame_cam
        elif point_img is not None:
            frame = point_img
        else:
            frame = None
        cv2.namedWindow('cam_name', 0)
        mmcv.imshow(frame, 'cam_name', wait_time=1)
        print(f"draw show time: {time.time() - start}")
        start = time.time()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg, args = parsing_args()
    main(cfg, args)
