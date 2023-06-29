# -*- coding: utf-8 -*-
# @Time: 2023/6/27 下午3:00
# @Author: YANG.C
# @File: common.py

from __future__ import annotations

import sys
import os

test_common_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(test_common_base)

from utils.log import logger, set_logger

from concurrency.job_package import JobPkgBase
from concurrency.thread_runner import ThreadRunner, ThreadMode


class JobInt(JobPkgBase):
    def __init__(self, number: int):
        super().__init__()
        self.number = number


class Detector(ThreadRunner):
    def __init__(self):
        super().__init__()

    def infer(self, job: JobInt) -> JobInt:
        self.ms_sleep(300)
        output = JobInt(job.number + 100)
        logger.info(f'Infer: {job.number} -> {output.number}')
        return output

    def add(self, job: JobInt) -> None:
        self.m_queueToWorker.put(job)

    def fetch(self) -> JobInt | None:
        if not self.m_queueFromWorker.empty():
            got: JobInt = self.m_queueFromWorker.get()
            logger.info(f'Detector output: {got.number}')
            return got
        else:
            return None

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.AllModes

    def _run_pre(self) -> None:
        logger.info(f'_run_pre')
        return

    def _run_pump(self) -> bool:
        logger.info(f'_run_pump')
        if not self.m_queueToWorker.empty():
            job_input = self.m_queueToWorker.get()
            if job_input is not None:
                self.m_queueFromWorker.put(self.infer(job_input))
        return True

    def _run_post(self) -> None:
        logger.info(f'_run_post')
        return


if __name__ == '__main__':
    set_logger(logger, './log.txt')
    detector = Detector()
    detector.m_brief_name = 'detector'
    detector._update_fullname()

    logger.info(f'supported mode: {detector.get_supported_mode()}')
    logger.info(f'current state pre start_run: {detector.get_state()}')

    detector.start_run()
    logger.info(f'current state after start_run: {detector.get_state()}')

    detector.add(JobInt(1))
    detector.add(JobInt(10))
    detector.add(JobInt(100))
