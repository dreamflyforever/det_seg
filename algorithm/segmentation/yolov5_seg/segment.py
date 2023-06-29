# -*- coding: utf-8 -*-
# @Time: 2023/6/28 下午5:02
# @Author: YANG.C
# @File: segment.py

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
from camera.capture import JobSharedImg

import algorithm.utils.yolo_postprocess as yolo_pp


class JobYOLOv5SegResult(JobPkgBase):
    """JobPkg of YOLOv5 detection results.

    Attributes:
        images: source image
        detections: ndarray with
    """

    def __init__(self, image: np.ndarray, detection: np.ndarray, masks: np.ndarray, key_frame: bool = True):
        super().__init__()
        self.image = image
        self.detections = detection
        self.key_frame = key_frame
        self.masks = masks


class YOLOv5SegmentWorker(BusWorker):
    """YOLOv5 Detector.

    Args:
        weight: path to exported yolov5 weight
        imgsz: input image size
        iou_thresh
        conf_thresh
        objectness
        max_det
    """

    def __init__(self, weight: str, imgsz: Union[Union[List, Tuple], int],
                 iou_thresh: float = 0.5, conf_thresh: float = 0.5, max_det: int = 300, ):
        super().__init__(ServiceId.DETECTOR, 'YOLOv5')
        assert Path(weight).expanduser().exists(), weight
        assert 0 <= iou_thresh < 1, f'Invalid iou threshold {conf_thresh}, it must range [0, 1)'
        assert 0 <= conf_thresh < 1, f'Invalid confidence threshold {conf_thresh}, it must range [0, 1)'

        self.weight = str(Path(weight).expanduser())
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.max_det = max_det

        self.onnx, self.rknn = [weight.endswith(x) for x in ('.onnx', '.rknn')]
        if not any((self.onnx, self.rknn)):
            raise ValueError(f'Not supported weight file: {self.weight}')

    def _load_model(self) -> None:
        if self.onnx:
            logger.info(f'model format is onnx')
            logger.info(f'{self.fullname()}, start loading model: {self.weight}')

            import onnxruntime as ort
            self.model = ort.InferenceSession(self.weight, None)
            self.onnx_input = self.model.get_inputs()[0].name
            self.onnx_outputs = [o.name for o in self.model.get_outputs()]

        else:
            logger.info(f'model format is rknn')
            logger.info(f'{self.fullname()}, start loading model: {self.weight}')

            from rknnlite.api import RKNNLite
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.weight)
            if ret != 0:
                logger.error(f'load RKNN model failed!')
                exit(ret)
            logger.info(f'Init runtime environment successful')

    def _warmup_model(self) -> None:
        assert self.model is not None, 'load model before warmup'
        logger.info(f'{self.fullname()}, start warmup model: {self.weight}')

        img = np.zeros((1, 3, *self.imgsz), dtype=np.float32)
        prev_time = 1e-9
        cnt = 0
        while cnt < 10:
            cnt += 1
            tick = time.time()
            if self.onnx:
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
        job_img = cast(JobSharedImg, self.m_queueToWorker.get())
        if job_img is None:
            return True

        if self.get_running_mode() == ThreadMode.InProcess:
            job_img.convert_to_shared_memory()
        else:
            job_img.convert_to_numpy()

        ori_img = job_img.copy_out()
        if not job_img.key_frame:
            job_detect_result = YOLOv5SegmentWorker(
                ori_img, np.empty((0, 6), np.float32)
            )
            job_detect_result.copy_tags(job_img)
            job_detect_result.key_frame = False
            self.m_queueFromWorker.put(job_detect_result)
            return False

        img, _ = self._preprocess(ori_img)
        img = img.astype('float32')
        img /= 255.
        if self.onnx:
            outputs = self.model.run(self.onnx_outputs, {self.onnx_input: img[None]})[0]
        else:
            outputs = self.model.inference(inputs=[img])
        pred, proto = outputs
        det, masks = self._postprocess(pred, proto, img.shape[:2], ori_img.shape[:2])
        job_segment_result = JobYOLOv5SegResult(ori_img.copy(), det, masks)
        job_segment_result.copy_tags(job_img)
        self.m_queueFromWorker.put(job_segment_result)

    def _preprocess(self, img: np.ndarray, color: Tuple[int, int, int] = (114, 114, 144)):
        img, _, [dw, dh] = yolo_pp.letterbox(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, [dw, dh]

    def _postprocess(self, pred: np.ndarray, proto: np.ndarray, img_shape: Tuple, im0_shape: Tuple):
        pred = yolo_pp.numpy_non_max_suppression(pred, self.conf_thresh, self.iou_thresh, max_det=self.max_det, nm=32)
        for i, det in enumerate(pred):
            if len(det):
                masks = yolo_pp.numpy_process_mask(proto[i], det[:, 6:], det[:, :4], img_shape, upsample=True)
                det[:, :4] = yolo_pp.scale_boxes(img_shape, det[:, :4], im0_shape).round()
        return det, masks

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes
