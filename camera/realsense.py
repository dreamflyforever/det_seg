# -*- coding: utf-8 -*-
# @Time: 2023/6/28 下午3:53
# @Author: YANG.C
# @File: realsense.py

from __future__ import annotations

import os
import sys
import time

realsense_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(realsense_base)

import math
import enum
import numpy as np
import cv2
import pyrealsense2 as rs
from shared_ndarray import SharedNDArray

from utils.log import logger
from camera.utils import CaptureState
from concurrency.bus import BusWorker, ServiceId, BusService
from concurrency.job_package import JobPkgBase
from concurrency.safe_queue import QueueSettings, QueueType
from concurrency.thread_runner import ThreadRunner, ThreadMode

pipeline = None
align = None
prev = 0
now = int(time.time())


class JobSharedRealsenseImg(JobPkgBase):
    def __init__(self, image: np.ndarray | SharedNDArray, key_frame: bool = False):
        super().__init__()
        self._private_image: np.ndarray | SharedNDArray | None = image
        self.key_frame = key_frame
        self.intrins_params = None
        # width, height, ppx, ppy, fx, fy, model, [coeffs]
        self.depth_params = [0, 0, 0, 0, 0, 0, None, [0, 0, 0, 0, 0]]
        self.aligned_depth_frame = None
        self.depth_image = None


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


class RealsenseCapture(BusWorker):
    def __init__(self, name: str):
        super().__init__(ServiceId.REALSENSE_CAPTURE, name)
        self.m_eof = False
        self.m_cam_id = 0
        self.m_url = 0
        self.m_state = CaptureState.Fresh.value
        self.m_gpu = -1
        self.m_fps = 15
        self.m_width = -1
        self.m_height = -1
        self.m_frame_id = -1
        self.m_frame_count = -1
        self.m_sample_frame: np.ndarray | None = None
        self.config_out_queue(QueueSettings(QueueType.BASIC, 25, True))
        self.m_cap: cv2.VideoCapture | None = None
        self.m_skip_frame = 0
        # self.pipeline = None
        # self.align = None

    def config(self, url: str = '', cam_id: int = 0, gpu: int = -1) -> None:
        self.m_url = url
        self.m_cam_id = cam_id
        self.m_gpu = gpu
        self.m_eof = False

    def eof(self) -> bool:
        return self.m_eof

    def collect_info(self, keep_open=False) -> None:
        self.set_realsense(width=640, height=360, fps=15)
        self.get_aligned_images()
        logger.info(f'{self.fullname()}, collect_info...')

    def _run_pre(self) -> None:
        self.m_eof = False
        self.collect_info(keep_open=True)
        if (pipeline and align) is None:
            logger.error(f'{self.fullname()}, Failed to open video file [{self.m_url}]')

    def _run_pump(self) -> bool:
        if self.m_eof:
            return True  # take a breath on EOF
        if not self.m_queueFromWorker.empty():
            return True  # take a breath when the out queue is not clear
        success = True
        if (pipeline and align) is not None:
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = self.get_aligned_images()
            capture_time = time.time() * 1e3
            if not depth_image.any() or not color_image.any() or not aligned_depth_frame:
                logger.error(f'{self.fullname()}, read all zeros depth or color image or aligned from realsense')
                success = False
        else:
            logger.error(f'{self.fullname()}, read depth or color image from realsense failed!')
            success = False

        if not success:
            # self.m_eof = True
            return True  # also breath
        
        # global now, prev
        # now = int(time.time())
        # if now > prev + 3:
        #     prev = now
        #     print(int(time.time()))
        #     cv2.imwrite(f'sample/{int(time.time())}.jpg', color_image)

        self.m_frame_id += 1
        finished_job = JobSharedRealsenseImg(color_image)
        finished_job.depth_image = depth_image
        finished_job.depth_params = [depth_intrin.width, depth_intrin.height, depth_intrin.ppx, depth_intrin.ppy,
                                    depth_intrin.fx, depth_intrin.fy, None, depth_intrin.coeffs]

        # logger.debug(f'aligned: {aligned_depth_frame.get_distance(100, 120)}')
        # logger.debug(f'distance: {depth_image[120][100]}, {depth_image[100][120]}')
        finished_job.time_stamp = capture_time

        # logger.info(f'{self.fullname()}, read image successful, {color_image.shape}')
        finished_job.frame_id = self.m_frame_id
        if self.m_frame_id % (self.m_skip_frame + 1) == 0:
            finished_job.key_frame = True
        if self.get_running_mode() == ThreadMode.InProcess:
            finished_job.convert_to_shared_memory()

        # logger.info(f'{self.fullname()}, finished_job: {vars(finished_job)}')
        # logger.info(f'{self.fullname()}, finished_job: {finished_job._private_image.shape}')
        self.m_queueFromWorker.put(finished_job)

        return False  # no breath coz there could be more incoming data

    def _run_post(self) -> None:
        self.m_eof = True
        self.m_frame_id = -1
        return

    def set_skip_frame(self, skip_frame):
        assert skip_frame >= 0, "The number if skip frame must be greater than or equal to 0"
        self.m_skip_frame = skip_frame

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes

    def set_realsense(self, width=640, height=360, fps=15):
        global pipeline, align
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        # prefetch one image
        self.get_aligned_images()

    def get_aligned_images(self):  # 逐帧从realsense读取图像
        time.sleep(5 / 1000.)
        frames = pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
        ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                             'ppx': intr.ppx, 'ppy': intr.ppy,
                             'height': intr.height, 'width': intr.width,
                             'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                             }'''

        # 保存内参到本地
        # with open('./intrinsics.json', 'w') as fp:
        # json.dump(camera_parameters, fp)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
        depth_image_3d = np.dstack(
            (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
        color_image = np.asanyarray(color_frame.get_data())  # RGB图

        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
