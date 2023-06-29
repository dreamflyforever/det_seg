# -*- coding: utf-8 -*-
# @Time: 2023/6/27 下午4:44
# @Author: YANG.C
# @File: bus.py

from __future__ import annotations

import os
import sys

bus_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(bus_base)

import enum
from abc import ABC
from typing import TypeVar

from concurrency.job_package import JobPkgBase
from concurrency.safe_queue import SafeQueue, QueueSettings, QueueType
from concurrency.thread_runner import ThreadRunner

J = TypeVar('J', bound=JobPkgBase)


class ServiceId(enum.IntEnum):
    NULL_MODULE = 0
    DETECTOR = 1
    SEGMENTATION = 2
    FILE_CAPTURE = 100

    @staticmethod
    def all_ids() -> tuple:
        return tuple(map(int, ServiceId))

    @staticmethod
    def all_items() -> dict:
        return {repr(i.name): int(i) for i in ServiceId}


class JobBus:
    def __init__(self):
        self.m_queue: {int: SafeQueue} = {sid: SafeQueue(n) for n, sid in ServiceId.all_items().items()}


g_PendingBus = JobBus()
g_CompleteBus = JobBus()


class BusService:
    # for main to call: to check if any completed job available
    @staticmethod
    def has_complete_job_fetch(sid: int) -> bool:
        return not g_CompleteBus.m_queue[sid].empty()

    @staticmethod
    def send_job_to_worker(sid: int, job: JobPkgBase) -> None:
        return g_PendingBus.m_queue[sid].add(job)

    @staticmethod
    def fetch_complete_job(sid: int) -> J:
        return g_CompleteBus.m_queue[sid].fetch()

    # for main to call: to get job-request sent from Main to workers
    @staticmethod
    def take_job(sid: int) -> J:
        return g_PendingBus.m_queue[sid].fetch()

    # for main to call: to check if any job waiting for execution
    @staticmethod
    def finish_job(sid: int, job: JobPkgBase) -> None:
        g_CompleteBus.m_queue[sid].add(job)

    @staticmethod
    def has_job_todo(sid: int) -> bool:
        return not g_PendingBus.m_queue[sid].empty()

    # get the queue from Main to worker
    @staticmethod
    def get_queue_app_to_worker(sid: int) -> SafeQueue:
        return g_PendingBus.m_queue[sid]

    # get the queue from worker to Main
    @staticmethod
    def get_queue_worker_to_app(sid: int) -> SafeQueue:
        return g_CompleteBus.m_queue[sid]

    # clear the two queues between Main and workers
    # call this method before you re-start a worker
    @staticmethod
    def clear_queue(sid: int) -> None:
        BusService.get_queue_worker_to_app(sid).clear()
        BusService.get_queue_app_to_worker(sid).clear()


# define a worker that pass jobs to/from the global bus or buses
class BusWorker(ThreadRunner, ABC):
    def __init__(self, service_id: ServiceId, name: str):
        super().__init__()
        self.m_service_id = service_id
        self.set_brief_name(name)
        self.config_in_queue(QueueSettings())
        self.config_out_queue(QueueSettings())

    def get_service_id(self) -> ServiceId:
        return self.m_service_id

    def config_in_queue(self, settings: QueueSettings) -> None:
        BusService.get_queue_app_to_worker(self.m_service_id).config(settings)

    def config_out_queue(self, settings: QueueSettings) -> None:
        BusService.get_queue_worker_to_app(self.m_service_id).config(settings)

    def pump(self) -> None:
        self._sync_with_buses()
        super(BusWorker, self).pump()

    def _sync_with_buses(self) -> None:
        if self.m_queueToWorker.empty():
            # transfer new job from Pending Queue to this worker
            job = BusService.take_job(self.m_service_id)
            if job is not None:
                self.m_queueToWorker.put(job)

        if not self.m_queueFromWorker.empty():
            # transfer completed job from this worker to Complete Queue
            q = BusService.get_queue_worker_to_app(self.m_service_id)
            qs = q.get_settings()
            need_do = True
            if qs.queue_type == QueueType.BASIC and q.full():
                need_do = False
            if need_do:
                job = self.m_queueFromWorker.get()
                if job is not None:
                    BusService.finish_job(self.m_service_id, job)
