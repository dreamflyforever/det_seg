# -*- coding: utf-8 -*-
# @Time: 2023/6/27 上午11:25
# @Author: YANG.C
# @File: test_queue_basic.py

from __future__ import annotations

import os
import sys

test_queue_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(test_queue_base)

from concurrency.safe_queue import SafeQueue, QueueType, QueueSettings
from concurrency.job_package import JobPkgBase


class JobInt(JobPkgBase):
    def __init__(self, number: int):
        super().__init__()
        self.number = number


def test_basic_drop():
    q = SafeQueue()
    settings = QueueSettings(QueueType.BASIC, 3, True)
    q.config(settings)

    # add 4 numbers to overflow
    q.add(JobInt(1))
    q.add(JobInt(5))
    q.add(JobInt(7))
    q.add(JobInt(9))

    # got only 3 numbers from queue
    assert q.fetch().number == 5
    assert q.fetch().number == 7
    assert q.fetch().number == 9
    assert q.fetch() is None


def test_basic_no_drop():
    q = SafeQueue()
    settings = QueueSettings(QueueType.BASIC, 3, False)
    q.config(settings)

    # add 4 numbers to overflow
    q.add(JobInt(1))
    q.add(JobInt(5))
    q.add(JobInt(7))
    q.add(JobInt(9))

    # got 4 numbers from queue
    assert q.fetch().number == 1
    assert q.fetch().number == 5
    assert q.fetch().number == 7
    assert q.fetch().number == 9
    assert q.fetch() is None
