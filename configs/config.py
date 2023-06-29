# -*- coding: utf-8 -*-
# @Time: 2023/6/27 上午9:25
# @Author: YANG.C
# @File: config.py

import sys
import os
import yaml
import time

cfg_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cfg_base)

f = open(f'{cfg_base}/configs/config.yaml', 'r')
cfg = yaml.load(f, Loader=yaml.FullLoader)

start_time = time.time()
ymd_time = time.localtime(start_time)
bj_time = time.strftime("%Y-%m-%d-%H-%M-%S", ymd_time)
cfg['log_file'] = f'{cfg["log"]["log_dir"]}/{bj_time}'

log_dir = cfg["log"]["log_dir"]
log_file = cfg['log_file']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
