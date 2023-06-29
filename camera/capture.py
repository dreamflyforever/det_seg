# -*- coding: utf-8 -*-
# @Time: 2023/6/28 上午10:56
# @Author: YANG.C
# @File: capture.py

from __future__ import annotations

import os
import sys

capture_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(capture_base)

import enum
import cv2
import numpy as np
from shared_ndarray import SharedNDArray

from utils.log import logger
from concurrency.bus import BusWorker, ServiceId, BusService
from concurrency.job_package import JobPkgBase
from concurrency.safe_queue import QueueSettings, QueueType
from concurrency.thread_runner import ThreadMode


class JobSharedImg(JobPkgBase):
    def __init__(self, image: np.ndarray | SharedNDArray, key_frame: bool = False):
        super().__init__()
        self._private_image: np.ndarray | SharedNDArray | None = image
        self.key_frame = key_frame

    def convert_to_shared_memory(self):
        assert self._private_image is not None
        if isinstance(self._private_image, np.ndarray):
            # do nothing if the image is already a SharedNDArray
            self._private_image = SharedNDArray.copy(self._private_image)

    def copy_out(self) -> np.ndarray | None:
        if isinstance(self._private_image, np.ndarray):
            # return ndarray
            out = self._private_image
            self._private_image = None
        elif isinstance(self._private_image, SharedNDArray):
            # convert SharedNDArray to ndarray
            out = np.copy(self._private_image)
            self._private_image.unlink()
            self._private_image = None
        else:
            out = None
        return out

    def convert_to_numpy(self):
        assert self._private_image is not None
        if isinstance(self._private_image, SharedNDArray):
            self._private_image = self.copy_out()

    def release(self):
        if self._private_image is None:
            return
        if isinstance(self._private_image, SharedNDArray):
            self._private_image.unlink()
            self._private_image = None
        elif isinstance(self._private_image, np.ndarray):
            self._private_image = None

    def __del__(self):
        self.release()
        pass


class CaptureState(enum.IntEnum):
    Fresh = 0x0
    Configured = 0x1
    Opened = 0x2
    Error = 0x8


class VideoFileCapture(BusWorker):
    def __init__(self, name: str):
        super().__init__(ServiceId.FILE_CAPTURE, name)
        self.set_brief_name('VfCap')
        self.m_eof = False
        self.m_cam_id = 0
        self.m_url = 0
        self.m_state = CaptureState.Fresh.value
        self.m_gpu = -1
        self.m_fps = 25
        self.m_width = -1
        self.m_height = -1
        self.m_frame_id = -1
        self.m_frame_count = -1
        self.m_sample_frame: np.ndarray | None = None
        self.config_out_queue(QueueSettings(QueueType.BASIC, 25, True))
        self.m_cap: cv2.VideoCapture | None = None
        self.m_skip_frame = 0

    # gpu also sames as rknn
    def config(self, url: str = '', cam_id: int = 0, gpu: int = -1) -> None:
        self.m_url = url
        self.m_cam_id = cam_id
        self.m_gpu = gpu
        self.m_eof = False

    def eof(self) -> bool:
        return self.m_eof

    def collect_info(self, keep_open=False) -> None:
        self.m_cap = cv2.VideoCapture(self.m_url)
        if self.m_cap.isOpened():
            self.m_frame_count = self.m_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.m_width = self.m_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.m_height = self.m_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.m_fps = self.m_cap.get(cv2.CAP_PROP_FPS)
            self.m_sample_frame = self.m_cap.read()
            BusService.get_queue_worker_to_app(self.m_service_id)
            if not keep_open:
                self.m_cap = self.m_cap.release()
        else:
            self.m_cap = None

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes

    def _run_pre(self) -> None:
        self.m_eof = False
        self.collect_info(keep_open=True)
        if self.m_cap is None:
            logger.error(f'{self.fullname()}, Failed to open video file [{self.m_url}]')

    def _run_pump(self) -> bool:
        if self.m_eof:
            return True  # take a breath on EOF
        if not self.m_queueFromWorker.empty():
            return True  # take a breath when the out queue is not clear
        if isinstance(self.m_cap, cv2.VideoCapture):
            if not self.m_cap.isOpened():
                self.m_eof = True
                return True  # take a breath on EOF
            success, img = self.m_cap.read()
        else:
            success, img = self.m_cap.nextFrame()
        if not success or img is None:
            self.m_eof = True
            return True  # also breath
        self.m_frame_id += 1
        finished_job = JobSharedImg(img)
        finished_job.frame_id = self.m_frame_id
        if self.m_frame_id % (self.m_skip_frame + 1) == 0:
            finished_job.key_frame = True
        if self.get_running_mode() == ThreadMode.InProcess:
            finished_job.convert_to_shared_memory()
        self.m_queueFromWorker.put(finished_job)
        return False  # no breath coz there could be more incoming data

    def _run_post(self) -> None:
        if self.m_cap is not None:
            if isinstance(self.m_cap, cv2.VideoCapture):
                self.m_cap.release()
            self.m_cap = None
            self.m_eof = True
            self.m_frame_id = -1
        return

    def set_skip_frame(self, skip_frame):
        assert skip_frame >= 0, "The number if skip frame must be greater than or equal to 0"
        self.m_skip_frame = skip_frame
