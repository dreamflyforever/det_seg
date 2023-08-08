# -*- coding: utf-8 -*-
# @Time: 2023/6/28 上午10:56
# @Author: YANG.C
# @File: capture.py

from __future__ import annotations

import os
import sys
import time

capture_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(capture_base)

import enum
import cv2
import numpy as np
from shared_ndarray import SharedNDArray

from utils.log import logger
from camera.utils import CaptureState
from concurrency.bus import BusWorker, ServiceId, BusService
from concurrency.job_package import JobPkgBase
from concurrency.safe_queue import QueueSettings, QueueType
from concurrency.thread_runner import ThreadMode


class JobSharedCapImg(JobPkgBase):
    def __init__(self, image: np.ndarray | SharedNDArray, key_frame: bool = False):
        super().__init__()
        self._private_image: np.ndarray | SharedNDArray | None = image
        self.key_frame = key_frame
        self.intrins_params = None

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
            out = np.copy(self._private_image.array)
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
        # self.release()
        pass


class VideoCapture(BusWorker):
    def __init__(self, name: str, hub: int = 0, width: int = 1600, height: int = 1200, fps: int = 25):
        super().__init__(ServiceId.FISH_CAPTURE, name)
        self.m_eof = False
        self.m_cam_id = 0
        self.m_url = 0
        self.m_hub = hub
        self.m_width = width
        self.m_height = height
        self.m_fps = fps
        self.m_state = CaptureState.Fresh.value
        self.m_gpu = -1
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
        self.m_cap = cv2.VideoCapture(self.m_hub + cv2.CAP_ANY)
        self.m_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.m_width)
        self.m_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.m_height)
        self.m_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.m_cap.set(cv2.CAP_PROP_FPS, self.m_fps)

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
        capture_time = time.time() * 1e3

        if not success or img is None:
            self.m_eof = True
            return True  # also breath

        self.m_frame_id += 1

        intrins_params, undistorted_img = self.fish_cam_undistort(img)
        finished_job = JobSharedCapImg(undistorted_img)
        finished_job.frame_id = self.m_frame_id
        finished_job.intrins_params = intrins_params
        finished_job.time_stamp = capture_time
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

    def fish_cam_undistort(self, img, scale=1.0, imshow=False, zc=True):
        k, d, dim = None, None, None
        if zc:
            # k = np.array(
            #     [[363.98953227211905, 0.0, 834.3158978931784], [0.0, 363.9524177963977, 650.107887396256],
            #      [0.0, 0.0, 1.0]])
            # d = np.array(
            #     [[0.002653439135043621], [-0.010520573385910588], [0.00144632201051458], [-0.0008507916189458718]])
	    k = np.array([[812.0759926090558, 0.0, 810.3963949199671], [0.0, 813.5716912856911, 606.0612337205697], [0.0, 0.0, 1.0]])
            d = np.array([[0.5033714773685056], [0.12237339795797722], [-0.41302582292485795], [0.4501049351340188]])
            dim = (1600, 1200)

        dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        # assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if dim1[0] != dim[0]:
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        knew = k.copy()
        if scale:  # change fov
            knew[(0, 1), (0, 1)] = scale * knew[(0, 1), (0, 1)]

        fx = knew[0][0]
        fy = knew[1][1]
        cx = knew[0][2]
        cy = knew[1][2]
        intrins_params = [fx, fy, cx, cy]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), knew, dim, cv2.CV_16SC2)
        undistorted = False
        if undistorted:
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            undistorted_img = img
        if imshow:
            cv2.imshow("undistorted", undistorted_img)
        return intrins_params, undistorted_img
