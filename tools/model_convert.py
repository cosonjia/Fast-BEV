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
# @brief  model_convert 
# @author coson.jia@evas.ai
# @date : 2023/3/23 下午3:50
# @file : model_convert.py
#
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import argparse
from functools import partial
def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument("--out-model", type=str)
    # parser.add_argument("--type", type=float, default='onnx')
    args = parser.parse_args()
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    return cfg, args

def export_onnx(model, dataset):
    import torch
    model.to('cuda:0')
    model.eval()
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        # return_loss=False,
        export_2d=False,
        export_3d=False,
        rescale=False)
    for data in dataset:
        img = data['img'].data[0].to('cuda:0')
        img_metas = data['img_metas'].data[0]
        img_metas[0].pop('box_type_3d')
        img_metas[0].pop('box_mode_3d')
        with torch.no_grad():
            torch.onnx.export(
                model,
                {'img':img,'img_metas':img_metas},
                f='tmp.onnx',
                input_names=['input'],
                output_names=None,
                export_params=True,
                keep_initializers_as_inputs=False,
                do_constant_folding=True,
                verbose=True,
                opset_version=16,
                dynamic_axes=None
            )
        break

def main(cfg, args):
    from mmdet3d.datasets import build_dataset, build_dataloader
    cfg_test = cfg.data.test
    pipeline = cfg_test.pipeline
    for p in pipeline:
        if p.type == 'LoadAnnotations3D':
                break
    pipeline.remove(p)
    dataset = build_dataset(cfg_test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    from mmdet3d.models import build_model
    model = build_model(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
    from mmcv.runner import load_checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    export_onnx(model,data_loader)
if __name__ == "__main__":
    cfg, args = parsing_args()
    main(cfg, args)
