# -*- coding: utf-8 -*-
# @Time: 2023/6/28 上午11:50
# @Author: YANG.C
# @File: yolo_handle.py

import os
import sys

yolo_pp_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(yolo_pp_base)

import numpy as np
import math
import time
import cv2

from utils.log import logger

# device tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def numpy_non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    if isinstance(prediction,
                  (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # x = torch.tensor(x)
        # conf, j = x[:, 5:mi].max(axis=1, keepdims=True)
        conf = np.amax(x[:, 5:mi], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:mi], axis=1).reshape(-1, 1)

        # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.reshape(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
        idx_desc = np.argsort(-x[:, 4])
        x = x[idx_desc][:max_nms]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        i = numpy_nms(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            logger.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def numpy_nms(dets, scores, thresh):
    order = scores.argsort()[::-1]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def numpy_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numpy_crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [n, h, w] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = np.array_split(boxes[:, :, None], 4, axis=1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def numpy_process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = numpy_sigmoid(masks_in @ protos.astype(np.float32).reshape(c, -1)).reshape(-1, mh, mw)  # CHW

    downsampled_bboxes = np.copy(bboxes)
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = numpy_crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = np.transpose(masks, (1, 2, 0))  # CHW -> HWC
        masks = cv2.resize(masks, shape, interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, np.newaxis]
        masks = np.transpose(masks, (2, 0, 1))  # HWC -> CHW
    return np.greater(masks, 0.5)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def find_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    centers = []
    rects = [cv2.minAreaRect(c) for c in contours]
    for rect in rects:
        box = cv2.boxPoints(rect).astype(np.int32)
        bboxes.append(box)
        center_x, center_y = np.mean(box, axis=0)
        centers.append([int(center_x), int(center_y)])
    return bboxes, centers


def calc_angle(bbox):
    # upper-left, upper-right, down-right, down-left
    p_ul, p_ur, p_dr, p_dl = bbox
    dist_width = np.linalg.norm(np.array(p_ul) - np.array(p_ur))
    dist_height = np.linalg.norm(np.array(p_ul) - np.array(p_dl))
    x_u = (p_ul[0] + p_ur[0]) / 2 if dist_width < dist_height else (p_ul[0] + p_dl[0]) / 2
    y_u = (p_ul[1] + p_ur[1]) / 2 if dist_width < dist_height else (p_ul[1] + p_dl[1]) / 2
    x_d = (p_dr[0] + p_dl[0]) / 2 if dist_width < dist_height else (p_ur[0] + p_dr[0]) / 2
    y_d = (p_dr[1] + p_dl[1]) / 2 if dist_width < dist_height else (p_ur[1] + p_dr[1]) / 2

    if x_u == x_d:
        yaw = 90
    else:
        dx, dy = x_u - x_d, y_u - y_d
        yaw = np.degrees(np.arctan2(abs(dy), abs(dx)))
        yaw = -yaw if dx * dy < 0 else yaw
        # calc yaw based on y axis
        yaw = 90 - yaw if yaw >= 0 else -(90 + yaw)
    return yaw


def seg_process(colors, det, masks, im, shape, alpha=0.5, visual=True):
    vis_colors = [colors(x, True) for x in det[:, 5]]
    vis_colors = np.array(vis_colors, dtype=np.float32) / 255.0
    vis_colors = vis_colors[:, np.newaxis, np.newaxis]

    new_masks = np.transpose(masks, (1, 2, 0))
    masks = masks[:, :, :, np.newaxis]
    masks_color = masks * (vis_colors * alpha)
    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

    im = im * inv_alph_masks[-1] + mcs
    im = im[:, :, ::-1]
    im_mask = (im * 255).astype(np.uint8)

    # resize mask to original size
    raw_masks = scale_image(im.shape, new_masks.astype(np.float32), shape)

    boxes = []
    yawes = []
    centers = []
    mask_nums = raw_masks.shape[2]
    for i in range(mask_nums):
        box, center = find_contours(raw_masks[:, :, i].astype('uint8') * 255)
        boxes.extend(box)
        centers.extend(center)

    vis_img = scale_image(im.shape, im_mask, shape)

    for box, center in zip(boxes, centers):
        yaw = calc_angle(box)
        yawes.append(yaw)

        if visual:
            cv2.drawContours(vis_img, [box], 0, (255, 255, 255), 2)

            theta_rad = math.radians(yaw)
            length = 70
            x0, y0 = center[0], center[1]
            x1 = int(x0 - length * math.cos(theta_rad))
            y1 = int(y0 - length * math.sin(theta_rad))
            x2 = int(x0 + length * math.cos(theta_rad))
            y2 = int(y0 + length * math.sin(theta_rad))

            text = str(int(yaw))
            cv2.putText(vis_img, text, (x0 + 10, y0 + 10), FONT, FONT_SCALE, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(vis_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    return yawes, centers, vis_img

def det_process(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        logger.debug(f'class: {int(cl)}, score: {float(score):.3f}')
        logger.debug(f'box coordinate left, top, right, down: [{top:.3f}, {left:.3f}, {right:.3f}, {bottom:.3f}]')
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(str(int(cl)), float(score)),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        return image