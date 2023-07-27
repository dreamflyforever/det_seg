# -*- coding: utf-8 -*-
# @Time: 2023/6/28 下午12:02
# @Author: YANG.C
# @File: detector.py

import os
import sys

detector_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(detector_base)

import time
from collections import namedtuple, OrderedDict
from pathlib import Path
from typing import Union, List, Tuple, cast

import cv2
import numpy as np

from utils.log import logger
from concurrency.bus import BusWorker, ServiceId
from concurrency.job_package import JobPkgBase
from concurrency.thread_runner import ThreadMode
from camera.realsense import JobSharedRealsenseImg

import algorithm.utils.yolo_handle as yolo_handle


class JobYOLOv5DetResult(JobPkgBase):
    """JobPkg of YOLOv5 detection results.

    Attributes:
        images: source image
        detections: ndarray with
    """

    def __init__(self, image: np.ndarray, detection: np.ndarray, key_frame: bool = True):
        super().__init__()
        self.image = image
        self.aligned_depth_frame = None
        self.depth_image = None
        self.intrins_params = None
        self.depth_params = None
        self.detections = detection
        self.key_frame = key_frame


class YOLOv5DetectWorker(BusWorker):
    """YOLOv5 Detector.

    Args:
        weight: path to exported yolov5 weight
        imgsz: input image size
        iou_thresh
        conf_thresh
        objectness
        max_det
    """

    def __init__(self, weight: str, imgsz: Union[Union[List, Tuple], int], cuda: bool = False,
                 iou_thresh: float = 0.5, conf_thresh: float = 0.5, max_det: int = 300, ):
        super().__init__(ServiceId.DETECTOR, 'YOLOv5Det')
        assert Path(weight).expanduser().exists(), weight
        assert 0 <= iou_thresh < 1, f'Invalid iou threshold {conf_thresh}, it must range [0, 1)'
        assert 0 <= conf_thresh < 1, f'Invalid confidence threshold {conf_thresh}, it must range [0, 1)'

        self.weight = str(Path(weight).expanduser())
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.cuda = cuda
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.max_det = max_det

        self.pt, self.onnx, self.rknn = [weight.endswith(x) for x in ('.pt', '.onnx', '.rknn')]
        if not any((self.pt, self.onnx, self.rknn)):
            raise ValueError(f'Not supported weight file: {self.weight}')

    def _load_model(self) -> None:
        if self.pt:
            logger.info(f'{self.fullname()}, model format is pytorch')
            logger.info(f'{self.fullname()}, start loading model: {self.weight}')
            import torch
            self.model = torch.load(self.weight)

        elif self.onnx:
            logger.info(f'{self.fullname()}, model format is onnx')
            logger.info(f'{self.fullname()}, start loading model: {self.weight}')
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.cuda else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(self.weight, providers=providers)
            self.onnx_input = self.model.get_inputs()[0].name
            self.onnx_outputs = [o.name for o in self.model.get_outputs()]

        else:
            logger.info(f'{self.fullname()}, model format is rknn')
            logger.info(f'{self.fullname()}, start loading model: {self.weight}')

            from rknnlite.api import RKNNLite
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.weight)
            if ret != 0:
                logger.error(f'{self.fullname()}, load RKNN model failed!')
                exit(ret)
            logger.info(f'{self.fullname()}, load RKNN model successful')

            ret = self.model.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            if ret != 0:
                logger.error(f'{self.fullname()}, Init runtime environment failed!')
                exit(ret)
            logger.info(f'{self.fullname()}, Init runtime environment successful')

    def _warmup_model(self) -> None:
        assert self.model is not None, 'load model before warmup'
        logger.info(f'{self.fullname()}, start warmup model: {self.weight}')

        img = np.zeros((1, 3, *self.imgsz), dtype=np.float32)
        prev_time = 1e-9
        cnt = 0
        while cnt < 10:
            cnt += 1
            tick = time.time()
            if self.pt:
                self.model(img)
            elif self.onnx:
                self.model.run(self.onnx_outputs, {self.onnx_input: img})
            else:
                self.model.inference(inputs=[img[0]])
            this_time = time.time() - tick
            if abs(this_time - prev_time) / prev_time < 0.05:
                break
            prev_time = this_time
        logger.info(f'{self.fullname()}, finished warmup model in {cnt} iterations')

    def _run_pre(self) -> None:
        try:
            self._load_model()
            self._warmup_model()
            self.is_ready = True
        except ImportError as error:
            raise error

    def _run_pump(self) -> bool:
        if not self.m_queueFromWorker.empty():
            return True
        if self.m_queueToWorker.empty():
            return True
        job_img = cast(JobSharedRealsenseImg, self.m_queueToWorker.get())
        if job_img is None:
            print('job_img is None')
            return True

        if self.get_running_mode() == ThreadMode.InProcess:
            job_img.convert_to_shared_memory()
        else:
            job_img.convert_to_numpy()

        ori_img = job_img.copy_out()
        if not job_img.key_frame:
            job_detect_result = JobYOLOv5DetResult(
                ori_img, np.empty((0, 4), np.float32)
            )
            job_detect_result.copy_tags(job_img)
            job_detect_result.key_frame = False
            self.m_queueFromWorker.put(job_detect_result)
            return False

        img, _ = self._preprocess(ori_img)
        img = img.astype('float32')
        # TODO
        if self.pt:
            img /= 255.
            outputs = self.model(img)
        elif self.onnx:
            img /= 255.
            outputs = self.model.run(self.onnx_outputs, {self.onnx_input: np.transpose(img, (2, 0, 1))[None]})[0]
        else:
            outputs = self.model.inference(inputs=[img])[0]
            outputs = np.array(outputs)
            outputs = np.squeeze(outputs, axis=-1)

        dets_per_img = self._postprocess(outputs, img.shape[:2], ori_img.shape[:2])
        if len(dets_per_img) == 0:
            job_detect_result = JobYOLOv5DetResult(image=ori_img,
                                                   detection=np.empty((0, 4), np.float32))
        else:
            boxes, scores, cls = np.split(dets_per_img[0], [4, 1 + 4], axis=1)
            vis_img = yolo_handle.det_process(ori_img, boxes, scores, cls)
            job_detect_result = JobYOLOv5DetResult(image=vis_img,
                                                   detection=boxes)
        job_detect_result.depth_image = job_img.depth_image
        job_detect_result.depth_params = job_img.depth_params
        job_detect_result.copy_tags(job_img)
        self.m_queueFromWorker.put(job_detect_result)
        return False

    def _run_post(self) -> None:
        del self.model
        self.model = None
        self.is_ready = None

    def _preprocess(self, img: np.ndarray, color: Tuple[int, int, int] = (114, 114, 144)):
        img, _, [dw, dh] = yolo_handle.letterbox(img, self.imgsz, color)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, [dw, dh]

    def _postprocess(self, pred: np.ndarray, img_shape: Tuple, im0_shape: Tuple):
        pred = yolo_handle.numpy_non_max_suppression(pred, self.conf_thresh, self.iou_thresh,
                                                     max_det=self.max_det, nm=0)  # detect: nm=0!
        dets_per_img = []
        for i, det in enumerate(pred):
            if len(det):
                logger.info(f'det: {det}, img_shape: {img_shape}, im0_shape: {im0_shape}')
                det[:, :4] = yolo_handle.scale_boxes(img_shape, det[:, :4], im0_shape).round()
                dets_per_img.append(det)
        return dets_per_img

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes
