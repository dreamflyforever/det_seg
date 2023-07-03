# -*- coding: utf-8 -*-
# @Time: 2023/6/30 上午10:17
# @Author: YANG.C
# @File: zmq_conn.py

from __future__ import annotations

import os
import sys

from typing import List, cast
import math

import numpy as np

zmq_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(zmq_base)

import zmq

from utils.log import logger
from concurrency.job_package import JobPkgBase
from concurrency.safe_queue import SafeQueue
from concurrency.bus import BusWorker, BusService, ServiceId
from connection.ArmCamera_pb2 import BottlePose, ObjectPose
from connection.rgbd_proto_pb2 import TargetLocation


class JobZMQData(JobPkgBase):
    def __init__(self, det_time: int, det_frame_id: int, det_xyzs: List[list],
                 seg_time: int, seg_frame_id: int, seg_yawes: List[float | int], seg_centers: List[List[int, int]]):
        super().__init__()
        self.det_time = det_time
        self.det_frame_id = det_frame_id
        self.seg_time = seg_time
        self.seg_frame_id = seg_frame_id
        self.det_xyzs = det_xyzs
        self.seg_yawes = seg_yawes
        self.seg_centers = seg_centers
        self.intrins_params = None


class ZMQConnection(BusWorker):
    def __init__(self, name: str, addrs: dict):
        super().__init__(ServiceId.ZMQ, name)
        self.addrs = addrs
        self.sockets = {}
        self.context = zmq.Context()

        self.shot_dis = 0.0042
        self.robot_height = 0.290
        self.robot_length = 0.3915
        self.camera_dis = 0.0175

    def _run_pre(self) -> None:
        try:
            for k, v in self.addrs.items():
                # TODO, more precise check
                if len(v) is None:
                    logger.info(f'{self.fullname()}, the [{k}] task has invalid zmq config, please check it.')
                else:
                    self.sockets[k] = self.context.socket(zmq.PUB)
                    self.sockets[k].bind(v)
        except Exception as error:
            raise error

    def _run_pump(self) -> bool:
        if not self.m_queueFromWorker.empty():
            return True
        if self.m_queueToWorker.empty():
            return True
        job_data = cast(JobZMQData, self.m_queueToWorker.get())
        if job_data is None:
            return True

        # detection
        det_xyzs = job_data.det_xyzs
        det_msg = TargetLocation()
        det_msg.seq = job_data.frame_id
        det_msg.ts = job_data.time_stamp
        if len(det_xyzs) == 0:
            det_msg.x1, det_msg.y1, det_msg.z1 = 9999, 9999, 9999
            det_msg.x2, det_msg.y2, det_msg.z2 = 9999, 9999, 9999
            logger.info(f'{self.fullname()}, no object be found!')
        else:
            i = len(det_xyzs) - 1
            det_msg.x1 = -det_xyzs[i][0]
            det_msg.y1 = det_xyzs[i][2]
            det_msg.z1 = det_xyzs[i][1]
            loc_re = self.location_translation(-det_msg.x1, det_msg.y1, det_msg.z1, 0)
            det_msg.x2 = loc_re[0] - self.camera_dis
            det_msg.y2 = loc_re[1] + self.robot_length - self.shot_dis
            det_msg.z2 = loc_re[2] - self.robot_height

        serialized_det_msg = det_msg.SerializeToString()
        self.sockets['detect'].send(serialized_det_msg)
        logger.info(f'{self.fullname()}, zmq send detection message successful')

        vec1 = np.array([det_msg.x1, det_msg.y1, det_msg.z1])
        logger.info(f'{self.fullname()}, local position: {vec1}')

        # segmentation
        seg_yawes = job_data.seg_yawes
        seg_centers = job_data.seg_centers
        seg_intrins_params = job_data.intrins_params
        seg_msg = BottlePose()
        if len(seg_yawes) == 0:
            seg_msg.data_valid = 0
            logger.info(f'{self.fullname()}, no bottle be found!')
        else:
            count = -1
            for i, (yaw, center) in enumerate(zip(seg_yawes, seg_centers)):
                logger.info(f'{i} object, yaw: {yaw}, center: {center}')
                fx, fy, cx, cy = seg_intrins_params
                real_loc = self.pixe2real(center, fx, fy, cx, cy)
                count += 1
                seg_msg.seq = count
                seg_msg.data_valid = 1
                seg_msg.x, seg_msg.y, seg_msg.z = real_loc[0] / 1000, real_loc[1] / 1000, real_loc[2] / 1000
                seg_msg.roll, seg_msg.pitch, seg_msg.yaw = 0, 0, yaw
        serialized_seg_msg = seg_msg.SerializeToString()
        self.sockets['segment'].send(serialized_seg_msg)
        logger.info(f'{self.fullname()}, zmq send segmentation message successful')

    @staticmethod
    def location_translation(x, y, z, theta):
        transmatix = np.array(
            [[-1, 0, 0], [0, math.sin(theta), math.cos(theta)], [0, math.cos(theta), -math.sin(theta)]])
        loc = np.array([x, y, z])
        x = x * (-1)
        theta = math.radians(theta)
        y = y * math.sin(theta) + y * math.cos(theta)
        z = z * (math.cos(theta) - math.sin(theta))
        loc_re = []
        loc_re.append(x)
        loc_re.append(y)
        loc_re.append(z)
        return loc_re

    @staticmethod
    def pixe2real(p_in, fx, fy, cx, cy):
        h = 375
        p_out = [0, 0, 0]
        p_out[0] = ((p_in[0] - cx) / fx) * h
        p_out[1] = ((p_in[1] - cy) / fy) * h
        p_out[2] = 1.0 * h
        p_out = tuple(p_out)
        return p_out
