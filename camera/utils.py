# -*- coding: utf-8 -*-
# @Time: 2023/6/29 下午2:52
# @Author: YANG.C
# @File: utils.py

from __future__ import annotations

import os
import sys

capture_utils_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(capture_utils_base)

import enum
import cv2
import numpy as np
from shared_ndarray import SharedNDArray

from utils.log import logger
from concurrency.bus import BusWorker, ServiceId, BusService
from concurrency.job_package import JobPkgBase
from concurrency.safe_queue import QueueSettings, QueueType
from concurrency.thread_runner import ThreadMode


class CaptureState(enum.IntEnum):
    Fresh = 0x0
    Configured = 0x1
    Opened = 0x2
    Error = 0x8
