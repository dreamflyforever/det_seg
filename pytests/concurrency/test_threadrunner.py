# -*- coding: utf-8 -*-
# @Time: 2023/6/27 下午2:57
# @Author: YANG.C
# @File: test_threadrunner.py

from __future__ import annotations

import os
import sys

test_thread_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(test_thread_base)

import time
import pytest

from concurrency.safe_queue import QueueSettings
from concurrency.thread_runner import ThreadRunner, ThreadMode
from pytests.unit.test_common import JobInt, Detector


@pytest.mark.parametrize('mode', [ThreadMode.InProcess, ThreadMode.Threaded, ThreadMode.Async])
def test_inference(mode: ThreadMode):
    t = Detector()
    t.set_brief_name('Detector')
    t.set_running_model(mode)

    t.start_run()
    t.add(JobInt(1))
    t.add(JobInt(5))
    t.add(JobInt(9))

    # 2000ms to run
    stop_time = time.time_ns() + 2 * QueueSettings.DefaultNanoUnit

    expected = [101, 105, 109]

    while time.time_ns() < stop_time:
        t.pump()
        job = t.fetch()
        if job is not None:
            assert expected[0] == job.number
            expected = expected[1:]

        ThreadRunner.ms_sleep(100)

    t.stop_and_wait()
    assert len(expected) == 0
    t.clear()
