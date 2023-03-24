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
# @brief  local_eval 
# @author coson.jia@evas.ai
# @date : 2023/3/17 上午11:25
# @file : local_eval.py
#
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
config='../configs/fastbev/exp/paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.py'
# outputs = mmcv.load('../fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_full-nuscenes-out.pkl')
outputs = mmcv.load('../work_dir/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4-mini.pkl')
eval_kwargs = dict(interval=5)
cfg = Config.fromfile(config)
# print(cfg.data.test)
dataset = build_dataset(cfg.data.val)
res = dataset.evaluate(outputs)
print(res)
