# -*- coding: utf-8 -*-
# @Time: 2023/8/10 下午3:38
# @Author: YANG.C
# @File: zmq_cs_conn.py

from __future__ import annotations

import os
import sys
import time

from typing import List, cast
import math

import numpy as np
import cv2

zmq_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(zmq_base)

import zmq
import pyrealsense2 as rs

from utils.log import logger
from algorithm.detection.yolov5_det.detector import JobYOLOv5DetResult
from algorithm.segmentation.yolov5_seg.segmentor import JobYOLOv5SegResult
from concurrency.job_package import JobPkgBase
from concurrency.thread_runner import ThreadMode
from concurrency.safe_queue import SafeQueue
from concurrency.bus import BusWorker, BusService, ServiceId
from connection.rgbd_proto_pb2 import TargetLocation
from connection.rgbd_repeat_pb2 import pose_array
# from connection.ArmCamera_pb2 import BottlePose
from connection.DualArmCamera_pb2 import BottlePoses
from connection.Camera2Robot_pb2 import *
from connection.CameraBottle2Robot_pb2 import *

CSAGAIN = [True]


class JobCSZMQData(JobPkgBase):
    def __init__(self, recv_time: int, recv_event: int):
        super().__init__()
        self.recv_time = recv_time
        self.recv_event = recv_event


class ZMQCSConnection(BusWorker):
    def __init__(self, name: str, addrs: dict):
        super().__init__(ServiceId.ZMQ_CS, name)
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
                    if k == 'detect':
                        self.sockets[k] = self.context.socket(zmq.REP)
                        self.sockets[k].bind(v)
                        logger.info(f'{self.fullname()}, the [{k}] task valid, bond on [{v}]')
        except Exception as error:
            raise error

    def _run_pump(self) -> bool:
        global CSAGAIN
        if CSAGAIN[0]:
            logger.warning(f'{self.fullname()}, listen client...')
            recv = CameraBottle2RobotRequest()
            msg_request = self.sockets['detect'].recv(copy=True)
            CSAGAIN[0] = False
            recv.ParseFromString(msg_request)
            logger.info(f'recv event: {recv.event}')
            recv_time = time.time() * 1e3
            recv_event = recv.event
            event_job = JobCSZMQData(recv_time, recv_event)
            event_job.recv_time = recv_time
            self.m_queueFromWorker.put(event_job)
            time.sleep(5 / 1000)

    def send_data(self, job_data) -> bool:
        # if not self.m_queueFromWorker.empty():
        #     return True
        # if self.m_queueToWorker.empty():
        #     return True
        # job_data = self.m_queueToWorker.get()
        # if job_data is None:
        #     return True

        # detection
        if isinstance(job_data, JobYOLOv5DetResult):
            # job_data 有很多objects, 对于每一个object，可以选择其两邻域的深度，再做深度平均（去除散斑无法命中的点）
            detections = job_data.detections
            depth_frame = job_data.depth_image
            depth_intrin = self.pickle_camera_params(job_data.depth_params)
            # xyzs = self.pix2camera(detections, depth_frame, depth_intrin)
            xyzs = self.pix2camera_with_repair(detections, depth_frame, depth_intrin, points=5)
            det_xyzs = xyzs

            single_obj = False
            multi_obj = True
            multi_vec = []
            if single_obj:
                # det_msg = TargetLocation()
                det_msg = FindBottleReply()
                det_msg.seq = job_data.frame_id
                det_msg.ts = job_data.time_stamp
                if len(det_xyzs) == 0:
                    det_msg.x1, det_msg.y1, det_msg.z1 = 9999, 9999, 9999
                    det_msg.x2, det_msg.y2, det_msg.z2 = 9999, 9999, 9999
                    logger.info(f'{self.fullname()}, Detector: no object be found!')
                else:
                    i = len(det_xyzs) - 1
                    det_msg.x1 = -det_xyzs[i][0]
                    det_msg.y1 = det_xyzs[i][2]
                    det_msg.z1 = det_xyzs[i][1]
                    loc_re = self.location_translation(-det_msg.x1, det_msg.y1, det_msg.z1, 0)
                    det_msg.x2 = loc_re[0] - self.camera_dis
                    det_msg.y2 = loc_re[1] + self.robot_length - self.shot_dis
                    det_msg.z2 = loc_re[2] - self.robot_height

            if multi_obj:
                c = CameraBottle2RobotReply()
                c.error.flag = True
                det_msg = FindBottleReply()
                if len(det_xyzs) == 0:
                    c.seq = job_data.frame_id
                    c.ts = job_data.time_stamp
                    p = det_msg.poses.pose.add()
                    p.x1, p.y1, p.z1 = 9999, 9999, 9999
                    p.x2, p.y2, p.z2 = 9999, 9999, 9999
                    mark = c.data
                    mark.Pack(det_msg)
                    logger.info(f'{self.fullname()}, Detector: no object be found!')
                else:
                    img = job_data.image
                    for i in range(len(det_xyzs)):
                        c.seq = job_data.frame_id
                        c.ts = job_data.time_stamp
                        p = det_msg.poses.pose.add()
                        p.x1 = -det_xyzs[i][0]
                        p.y1 = det_xyzs[i][2]
                        p.z1 = det_xyzs[i][1]
                        loc_re = self.location_translation(-p.x1, p.y1, p.z1, 0)
                        p.x2 = loc_re[0] - self.camera_dis
                        p.y2 = loc_re[1] + self.robot_length - self.shot_dis
                        p.z2 = loc_re[2] - self.robot_height
                        multi_vec.append([p.x2, p.y2, p.z2])
                        mark = c.data
                        mark.Pack(det_msg)
                        text = f'{p.x2}, {p.y2}, {p.z2}'
                        cv2.putText(img, text, (int(detections[i][0]) + 20, int(detections[i][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.imwrite(f'obj/{time.time() * 1e3}.jpg', img)

            # if multi_obj:
            #     det_msg = pose_array()
            #     if len(det_xyzs) == 0:
            #         p = det_msg.pose.add()
            #         p.seq = job_data.frame_id
            #         p.ts = job_data.time_stamp
            #         p.x1, p.y1, p.z1 = 9999, 9999, 9999
            #         p.x2, p.y2, p.z2 = 9999, 9999, 9999
            #         logger.info(f'{self.fullname()}, Detector: no object be found!')
            #     else:
            #         for i in range(len(det_xyzs)):
            #             p = det_msg.pose.add()
            #             p.seq = job_data.frame_id
            #             p.ts = job_data.time_stamp
            #             p.x1 = -det_xyzs[i][0]
            #             p.y1 = det_xyzs[i][2]
            #             p.z1 = det_xyzs[i][1]
            #             loc_re = self.location_translation(-p.x1, p.y1, p.z1, 0)
            #             p.x2 = loc_re[0] - self.camera_dis
            #             p.y2 = loc_re[1] + self.robot_length - self.shot_dis
            #             p.z2 = loc_re[2] - self.robot_height
            #             multi_vec.append([p.x1, p.y1, p.z1])
            serialized_det_msg = c.SerializeToString()
            # logger.info(f'ser msg: {serialized_det_msg}')
            # c = CameraBottle2RobotReply()
            # c.seq = 1
            # c.error.flag = True
            # msg = c.SerializeToString()
            self.sockets['detect'].send(serialized_det_msg)
            # self.sockets['detect'].send(serialized_det_msg)

            if single_obj:
                result_vec = np.array([det_msg.x1, det_msg.y1, det_msg.z1])
            elif multi_obj:
                result_vec = np.array(multi_vec)

            logger.info(
                f'{self.fullname()}-Detection, zmq send detection message successful, single obj: {single_obj}, '
                f'multi obj: {multi_obj}, local position: {result_vec}')

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes

    def _run_post(self) -> None:
        for k, _ in self.sockets.items():
            del self.sockets[k]

    @staticmethod
    def pickle_camera_params(depth_intrin):
        depth_intrinsic = rs.intrinsics()
        depth_intrinsic.width = depth_intrin[0]
        depth_intrinsic.height = depth_intrin[1]
        depth_intrinsic.ppx = depth_intrin[2]
        depth_intrinsic.ppy = depth_intrin[3]
        depth_intrinsic.fx = depth_intrin[4]
        depth_intrinsic.fy = depth_intrin[5]
        depth_intrinsic.model = rs.distortion.inverse_brown_conrady
        depth_intrinsic.coeffs = depth_intrin[7]
        return depth_intrinsic

    @staticmethod
    def pix2camera(xyxy_list, aligned_depth_frame, depth_intrin):
        camera_xyz_list = []  # 坐标二维列表
        if len(xyxy_list) > 0:
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)  # 计算像素坐标系的x
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)  # 计算像素坐标系的y
                dis = aligned_depth_frame[uy][ux] / 1000.
                camera_xyz = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                camera_xyz = camera_xyz.tolist()
                camera_xyz = camera_xyz + [0, 0, 0]
                camera_xyz_list.append(camera_xyz)
        return camera_xyz_list

    @staticmethod
    def pix2camera_with_repair(xyxy_list, aligned_depth_frame, depth_intrin, points=3):
        # TODO: test latency for support more points
        assert points in [3, 5]
        camera_xyz_list = []  # 坐标二维列表
        # 40cm - 1cm, 100cm - 2cm
        if len(xyxy_list) > 0:
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)  # 计算像素坐标系的x
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)  # 计算像素坐标系的y
                # TODO: more general for many points
                if points == 3:
                    ux_points = [ux - 1, ux, ux + 1]
                    uy_points = [uy, uy, uy]
                else:
                    ux_points = [ux - 1, ux, ux + 1, ux, ux]
                    uy_points = [uy, uy, uy, uy - 1, uy + 1]
                x_points, y_points = [], []
                z_points = []
                # [(x-1, y), (x, y), (x+1, y)]
                for x, y in zip(ux_points, uy_points):
                    dis = aligned_depth_frame[y][x] / 1000.
                    if dis == 0:  # drop zero
                        continue
                    camera_xyz = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, (x, y), dis)  # 计算相机坐标系的xyz
                    camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                    camera_xyz = camera_xyz.tolist()
                    camera_xyz = camera_xyz + [0, 0, 0]
                    x_points.append(camera_xyz[0])
                    y_points.append(camera_xyz[1])
                    z_points.append(camera_xyz[2])

                if len(z_points):
                    z_avg = sum(z_points) / len(z_points)

                    offset = z_avg / 40  # 1m -> 2.5cm, 2m -> 5cm
                    filter_index = []
                    # filter
                    for i in range(len(z_points)):
                        if (z_points[i] - z_avg) > offset:
                            filter_index.append(i)
                    if len(filter_index) == len(z_points):
                        continue
                    x_sum, y_sum, z_sum = 0, 0, 0
                    for i in range(len(z_points)):
                        if i not in filter_index:
                            x_sum += x_points[i]
                            y_sum += y_points[i]
                            z_sum += z_points[i]
                    x_avg = x_sum / (len(x_points) - len(filter_index))
                    y_avg = y_sum / (len(y_points) - len(filter_index))
                    z_avg = z_sum / (len(z_points) - len(filter_index))
                    camera_xyz_list.append([x_avg, y_avg, z_avg])
        return camera_xyz_list

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
    def pix2real(p_in, fx, fy, cx, cy):
        h = 310 - 30
        p_out = [0, 0, 0]
        p_out[0] = ((p_in[0] - cx) / fx) * h
        p_out[1] = ((p_in[1] - cy) / fy) * h
        p_out[2] = 1.0 * h
        p_out = tuple(p_out)
        return p_out

    def pix2real_with_repair(self, p_in, fx, fy, cx, cy):
        h = 310 - 30
        p_out = [0, 0, 0]
        p_out[0] = ((p_in[0] - cx) / fx) * h
        p_out[1] = ((p_in[1] - cy) / fy) * h
        p_out[2] = 1.0 * h

        px, py = p_in
        offset_x = self.gaussian_compensation(px, u=800)
        offset_x = -offset_x if px < 800 else offset_x
        offset_y = self.gaussian_compensation(py, u=600)
        offset_y = -offset_y if px < 600 else offset_y
        p_out[0] += offset_x
        p_out[1] += offset_y
        p_out = tuple(p_out)
        return p_out

    @staticmethod
    def gaussian_compensation(x, u, sigma=300, offset_max=2.5):
        gauss = np.exp(-((x - u) ** 2) / (2 * sigma ** 2))
        offset = offset_max * gauss
        return offset_max - offset
