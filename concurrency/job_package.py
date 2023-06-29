# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:04
# @Author: YANG.C
# @File: jobpackage.py

from __future__ import annotations

import abc
import time


class JobPkgBase(abc.ABC):
    def __init__(self):
        self.camera_id = 0
        self.frame_id = 0
        self.object_id = 0
        self.time_stamp = 0
        self.valid_data = False
        self.flags = 0
        self.sync_id = 0
        self.reserved = 0

    def copy_tags(self, in_data: JobPkgBase) -> None:
        self.camera_id = in_data.camera_id
        self.frame_id = in_data.frame_id
        self.object_id = in_data.object_id
        self.time_stamp = in_data.time_stamp
        self.valid_data = in_data.valid_data
        self.sync_id = in_data.sync_id
        self.reserved = in_data.reserved

    def same_tags(self, in_data: JobPkgBase) -> bool:
        return self.camera_id == in_data.camera_id and self.frame_id == in_data.frame_id and self.object_id == \
            in_data.object_id and self.time_stamp == in_data.time_stamp

    def same_frame(self, in_data: JobPkgBase) -> bool:
        return self.time_stamp == in_data.time_stamp

    def same_camera(self, in_data: JobPkgBase) -> bool:
        return self.camera_id == in_data.camera_id

    def same_sequence(self, in_data: JobPkgBase) -> bool:
        return self.camera_id == in_data.camera_id and self.sync_id == in_data.sync_id

    def mark_job_complete(self, data_valid: bool) -> None:
        self.valid_data = data_valid
        self.time_stamp = time.time_ns()
