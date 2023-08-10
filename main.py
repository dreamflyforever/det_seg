# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:00
# @Author: YANG.C
# @File: main.py


import sys
import os
import time

sys.path.append('../pyrobot')

from typing import cast

from easydict import EasyDict
import cv2

from configs.config import cfg
from utils.log import logger, set_logger

from algorithm.segmentation.yolov5_seg.segmentor import YOLOv5SegmentWorker, JobYOLOv5SegResult
from algorithm.detection.yolov5_det.detector import YOLOv5DetectWorker, JobYOLOv5DetResult
from concurrency.bus import BusService, BusWorker
from concurrency.safe_queue import SafeQueue, QueueSettings, QueueType
from concurrency.thread_runner import ThreadRunner, ThreadMode
from camera.realsense import RealsenseCapture, JobSharedRealsenseImg
from camera.capture import VideoCapture, JobSharedCapImg
from connection.zmq_conn import ZMQConnection
from connection.zmq_cs_conn import ZMQCSConnection, JobCSZMQData, CSAGAIN

# debug
from concurrency.bus import g_PendingBus, g_CompleteBus


def main():
    configs = EasyDict(cfg)
    log_file = configs.log_file
    set_logger(logger, log_file)
    logger.debug('init logger system successful')
    logger.debug(configs)

    mode = QueueType.BASIC
    visualization = configs.visualization

    # load
    if configs.models.detect.valid:
        # load model
        det_weight = configs.models.detect.weight
        det_obj_thres = configs.models.detect.obj_threshold
        det_nms_thres = configs.models.detect.nms_threshold
        det_imgsz = configs.models.detect.imgsz
        detector = YOLOv5DetectWorker(det_weight, det_imgsz, iou_thresh=det_nms_thres, conf_thresh=det_obj_thres)
        detector.config_in_queue(QueueSettings(mode, 1, True))
        detector.config_out_queue(QueueSettings(mode, 1, True))
        did = detector.get_service_id()
        detector.set_running_model(ThreadMode.InProcess)
        detector.start_run()
        BusService.clear_queue(did)

        # pull flow
        realsense_cap = RealsenseCapture('RSCap')
        realsense_cap.config_out_queue(QueueSettings(mode, 1, True))
        realsense_cap.set_running_model(ThreadMode.Threaded)
        realsense_cap.config()
        realsense_cap.start_run()
        rid = realsense_cap.get_service_id()
        BusService.clear_queue(rid)

    # load segment
    if configs.models.segment.valid:
        seg_weight = configs.models.segment.weight
        seg_obj_thres = configs.models.segment.obj_threshold
        seg_nms_thres = configs.models.segment.nms_threshold
        seg_imgsz = configs.models.segment.imgsz
        segmentor = YOLOv5SegmentWorker(seg_weight, seg_imgsz, iou_thresh=seg_nms_thres, conf_thresh=seg_obj_thres)
        segmentor.config_in_queue(QueueSettings(mode, 1, True))
        segmentor.config_out_queue(QueueSettings(mode, 1, True))
        sid = segmentor.get_service_id()
        segmentor.set_running_model(ThreadMode.InProcess)
        segmentor.start_run()
        BusService.clear_queue(sid)

        fish_cap_hub = configs.cameras.fish.hub
        fish_cap_width = configs.cameras.fish.width
        fish_cap_height = configs.cameras.fish.height
        fish_cap_fps = configs.cameras.fish.fps
        fish_cap_k = configs.cameras.fish.k
        fish_cap_d = configs.cameras.fish.d
        fish_cap = VideoCapture('FVCap', hub=fish_cap_hub, width=fish_cap_width, height=fish_cap_height,
                                fps=fish_cap_fps, k=fish_cap_k, d=fish_cap_d)
        fish_cap.config_out_queue(QueueSettings(mode, 1, True))
        fish_cap.set_running_model(ThreadMode.Threaded)
        fish_cap.config()
        fish_cap.start_run()
        fid = fish_cap.get_service_id()
        BusService.clear_queue(fid)

    # create connection
    zmq_addrs = {}
    zmq_det_valid = False
    zmq_seg_valid = False
    detect_addr = configs.connections.detect.zmq.addr
    if len(detect_addr) > 0:
        zmq_addrs['detect'] = detect_addr
        zmq_det_valid = True
    segment_addr = configs.connections.segment.zmq.addr
    if len(segment_addr) > 0:
        zmq_addrs['segment'] = segment_addr
        zmq_seg_valid = True

    if len(zmq_addrs.keys()) > 0:
        # only segment
        pb_zmq = ZMQConnection('connections-segment', zmq_addrs)
        pb_zmq.set_running_model(ThreadMode.Threaded)
        pb_zmq.start_run()
        pb_zid = pb_zmq.get_service_id()
        BusService.clear_queue(pb_zid)

    if len(zmq_addrs.keys()) > 0:
        cs_zmq = ZMQCSConnection('connections-detect', zmq_addrs)
        cs_zmq.set_running_model(ThreadMode.Threaded)
        cs_zmq.start_run()
        cs_zid = cs_zmq.get_service_id()
        BusService.clear_queue(cs_zid)

    need_stop = False
    cs_recv = False
    while not need_stop:
        if configs.models.detect.valid:
            realsense_cap.pump()  # get a frame from realsense to BusServer(g_CompleteBus)
            detector.pump()
        if configs.models.segment.valid:
            fish_cap.pump()
            segmentor.pump()  # get a frame from BusServer()
        if configs.connections.valid:
            pb_zmq.pump()
            cs_zmq.pump()

        if configs.models.detect.valid:
            # from zmq
            if not BusService.get_queue_app_to_worker(cs_zid).full():
                recv_cs_job = cast(JobCSZMQData, BusService.fetch_complete_job(cs_zid))
                if recv_cs_job is not None or cs_recv:
                    cs_recv = True
                    if realsense_cap.m_eof and not BusService.has_complete_job_fetch(rid):
                        break
                    # read a frame from video when to detect pending queue is not full
                    if not BusService.get_queue_app_to_worker(rid).full():
                        # get a frame from BusServer(g_CompleteBus)
                        realsense_cap_job = cast(JobSharedRealsenseImg, BusService.fetch_complete_job(rid))
                        if realsense_cap_job is not None:
                            # send a frame to BusServer(g_PendingBus)
                            BusService.send_job_to_worker(did, realsense_cap_job)

                    det_job = cast(JobYOLOv5DetResult, BusService.fetch_complete_job(did))

                    if det_job is not None:
                        # send to zmq
                        if detect_addr is not None and zmq_det_valid:
                            # BusService.send_job_to_worker(zid, det_job)
                            cs_zmq.send_data(det_job)
                            cs_recv = False
                            CSAGAIN[0] = True

                        if visualization:
                            color_image = det_job.image
                            cv2.namedWindow('det', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                            cv2.imshow('det', color_image)
                            key = cv2.waitKey(1)
                            if key & 0xFF == ord('q') or key == 27:
                                cv2.destroyAllWindows()
                                break

        if configs.models.segment.valid:
            if fish_cap.m_eof and not BusService.has_complete_job_fetch(fid):
                break
            # read a frame from video when to detect pending queue is not full
            if not BusService.get_queue_app_to_worker(fid).full():
                fish_cap_job = cast(JobSharedCapImg, BusService.fetch_complete_job(fid))
                if fish_cap_job is not None:
                    BusService.send_job_to_worker(sid, fish_cap_job)

            # read segmentation results
            seg_job = cast(JobYOLOv5SegResult, BusService.fetch_complete_job(sid))
            if seg_job is not None:
                # send to zmq
                if segment_addr is not None and zmq_seg_valid:
                    BusService.send_job_to_worker(zid, seg_job)

                # logger.info(f'{seg_job}')
                if visualization:
                    color_image = seg_job.image
                    cv2.namedWindow('seg', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.imshow('seg', color_image)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q') or key == 27:  # 执行时，按q或esc退出
                        cv2.destroyAllWindows()
                        break


if __name__ == '__main__':
    # import cProfile
    #
    # cProfile.run("main()")
    main()
