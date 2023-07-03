# -*- coding: utf-8 -*-
# @Time: 2023/6/28 下午5:02
# @Author: YANG.C
# @File: segmentor.py

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
from camera.capture import JobSharedCapImg
import algorithm.utils.yolo_handle as yolo_handle


class JobYOLOv5SegResult(JobPkgBase):
    """JobPkg of YOLOv5 Segmentation results.

    Attributes:
        images: source image
        detections: ndarray with
    """

    def __init__(self, image: np.ndarray, detection: List[np.ndarray], masks: List[np.ndarray], key_frame: bool = True,
                 intrins_params=None, yawes=None, centers=None):
        super().__init__()
        self.image = image
        self.detections = detection
        self.key_frame = key_frame
        self.masks = masks
        self.intrins_params = intrins_params
        self.yawes = yawes
        self.centers = centers


class YOLOv5SegmentWorker(BusWorker):
    """YOLOv5 Segmentor.

    Args:
        weight: path to exported yolov5 weight
        imgsz: input image size
        iou_thresh
        conf_thresh
        objectness
        max_det
    """

    def __init__(self, weight: str = '', imgsz: Union[Union[List, Tuple], int] = 640, cuda: bool = False,
                 iou_thresh: float = 0.5, conf_thresh: float = 0.5, max_det: int = 300, ):
        super().__init__(ServiceId.SEGMENTATION, 'YOLOv5Seg')
        assert Path(weight).expanduser().exists(), weight
        assert 0 <= iou_thresh < 1, f'Invalid iou threshold {conf_thresh}, it must range [0, 1)'
        assert 0 <= conf_thresh < 1, f'Invalid confidence threshold {conf_thresh}, it must range [0, 1)'

        self.weight = str(Path(weight).expanduser())
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.cuda = cuda
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.max_det = max_det
        self.colors = yolo_handle.Colors()

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
            logger.info(f'{self.fullname()}, model format is RKNN')
            logger.info(f'{self.fullname()}, start loading model: {self.weight}')

            from rknnlite.api import RKNNLite
            self.model = RKNNLite()
            ret = self.model.load_rknn(self.weight)
            if ret != 0:
                logger.error(f'{self.fullname()}, load RKNN model failed!')
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
        job_img = cast(JobSharedCapImg, self.m_queueToWorker.get())
        if job_img is None:
            return True

        if self.get_running_mode() == ThreadMode.InProcess:
            job_img.convert_to_shared_memory()
        else:
            job_img.convert_to_numpy()

        ori_img = job_img.copy_out()
        if not job_img.key_frame:
            job_detect_result = JobYOLOv5SegResult(
                image=ori_img,
                detection=[np.empty((0, 6), np.float32)],
                masks=[np.empty((0, 6), np.float32)]
            )
            job_detect_result.copy_tags(job_img)
            job_detect_result.key_frame = False
            self.m_queueFromWorker.put(job_detect_result)
            return False

        img, _ = self._preprocess(ori_img)
        img = img.astype('float32')
        img /= 255.
        # TODO
        if self.pt:
            outputs = self.model(img)
        if self.onnx:
            outputs = self.model.run(self.onnx_outputs, {self.onnx_input: np.transpose(img, (2, 0, 1))[None]})
        else:
            outputs = self.model.inference(inputs=[img])  # RKNN
        pred, proto = outputs
        if self.rknn:
            pred = np.squeeze(pred, axis=-1)
        dets_per_img, masks_per_img = self._postprocess(pred, proto, img.shape[:2], ori_img.shape[:2])
        if len(dets_per_img):
            yawes, centers, vis_img = yolo_handle.seg_process(self.colors, dets_per_img[0], masks_per_img[0], img,
                                                              ori_img.shape)
            job_segment_result = JobYOLOv5SegResult(vis_img.copy(), dets_per_img, masks_per_img,
                                                    job_img.intrins_params, yawes, centers)

        else:
            job_segment_result = JobYOLOv5SegResult(ori_img.copy(), dets_per_img, masks_per_img, job_img.intrins_params)

        job_segment_result.copy_tags(job_img)
        self.m_queueFromWorker.put(job_segment_result)

    def _run_post(self) -> None:
        del self.model
        self.model = None
        self.is_ready = None

    def _preprocess(self, img: np.ndarray, color: Tuple[int, int, int] = (114, 114, 144)):
        img, _, [dw, dh] = yolo_handle.letterbox(img, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, [dw, dh]

    def _postprocess(self, pred: np.ndarray, proto: np.ndarray, img_shape: Tuple, im0_shape: Tuple):
        pred = yolo_handle.numpy_non_max_suppression(pred, self.conf_thresh, self.iou_thresh,
                                                     max_det=self.max_det, nm=32)  # segment: nm=32!
        dets_per_img = []
        masks_per_img = []
        for i, det in enumerate(pred):
            if len(det):
                masks = yolo_handle.numpy_process_mask(proto[i], det[:, 6:], det[:, :4], img_shape, upsample=True)
                det[:, :4] = yolo_handle.scale_boxes(img_shape, det[:, :4], im0_shape).round()
                dets_per_img.append(det)
                masks_per_img.append(masks)
        return dets_per_img, masks_per_img

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes


if __name__ == '__main__':
    yolo = YOLOv5SegmentWorker()
    yolo.set_running_model(ThreadMode.Threaded)
    yolo.start_run()

    print(yolo.m_thread, yolo.m_queueFromWorker)
    print(yolo.fullname())
