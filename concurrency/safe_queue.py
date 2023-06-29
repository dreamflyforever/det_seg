# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:13
# @Author: YANG.C
# @File: safe_queue.py

from __future__ import annotations

import os
import sys

queue_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(queue_base)

import threading
import time
from enum import IntEnum

from concurrency.instance_name import InstanceName
from concurrency.job_package import JobPkgBase
from utils.log import logger


class QueueType(IntEnum):
    BASIC = 1,  # almost all queue
    DUPLEX = 2,
    CLOCK = 3,  # timer
    SAMPLE = 4,  # capture
    ORDERED = 5,


class QueueSettings:
    DefaultLength = 25
    DefaultLatencyMs = 1e3
    DefaultNanoUnit = 1e9

    def __init__(self, queue_type=QueueType.BASIC, length=DefaultLength, drop=False):
        self.queue_type = queue_type
        self.length = length
        self.drop_on_full = drop
        self.latency_ms = QueueSettings.DefaultLatencyMs


class SafeQueue(InstanceName):
    def __init__(self, brief_name='No_name'):
        super().__init__(brief_name=brief_name)
        self.m_log_flag = True
        self.m_cached_size = 0
        self.m_sync_id = 0
        self.m_latest_timestamp = 0  # time point of last pop-up
        self.m_first_fetch_time = 0  # time point of first pop-up
        self.m_queue: list[JobPkgBase] = []  # each item must be of JobPkgBase type
        self.m_frame_count = 0

        self.m_settings = QueueSettings(QueueType.BASIC, 25, True)

        # mutex
        self.m_lock = threading.RLock()

    def size(self) -> int:
        return self.m_cached_size

    def full(self) -> bool:
        return self.size() >= self.m_settings.length

    def empty(self) -> bool:
        return self.size() == 0

    def get_settings(self) -> QueueSettings:
        return self.m_settings

    def clear(self) -> None:
        with self.m_lock:
            self.m_queue.clear()
            self.m_cached_size = 0
            self.m_sync_id = 0
            self.m_frame_count = 0
            self.m_latest_timestamp = 0
            self.m_first_fetch_time = 0

    def config_log_flag(self, flag: bool) -> None:
        with self.m_lock:
            self.m_log_flag = flag

    def config_type(self, queue_type: QueueType) -> None:
        with self.m_lock:
            self.m_settings.queue_type = queue_type

    def config(self, setting: QueueSettings) -> None:
        with self.m_lock:
            self.m_settings = setting

    def config_latency(self, latency_ms) -> None:
        with self.m_lock:
            self.m_settings.latency_ms = latency_ms

    def config_length(self, length: int) -> None:
        with self.m_lock:
            self.m_settings.length = length

    def config_drop(self, drop_on_full: bool) -> None:
        with self.m_lock:
            self.m_settings.drop_on_full = drop_on_full

    def debug_print(self):
        with self.m_lock:
            logger.debug(f'{self.fullname()}, size = {self.size()}')

    def add(self, in_data: JobPkgBase) -> None:
        assert in_data is not None
        with self.m_lock:
            self.m_queue.append(in_data)

            while self.m_settings.drop_on_full and len(self.m_queue) > self.m_settings.length:
                self.m_queue = self.m_queue[1:]  # drop queue front
                self.m_frame_count += 1
                if self.m_log_flag:
                    logger.debug(
                        f'{self.fullname()}, a data is dropped when adding to the full queue, size = {len(self.m_queue)}')
            self.m_cached_size = len(self.m_queue)

    def fetch(self) -> JobPkgBase | None:
        if self.m_settings.queue_type == QueueType.BASIC:
            return self.fetch_basic()

        assert False

    def fetch_basic(self) -> JobPkgBase | None:
        if self.empty():  # pre-check
            return None

        with self.m_lock:  # check again when you want to fetch
            if self.empty():
                return None

            out_data = self.m_queue[0]
            self.m_queue = self.m_queue[1:]
            self.m_frame_count += 1
            self.m_cached_size = len(self.m_queue)
            return out_data
