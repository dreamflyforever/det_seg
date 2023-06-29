# -*- coding: utf-8 -*-
# @Time: 2023/6/28 上午10:23
# @Author: YANG.C
# @File: test_bus.py

from __future__ import annotations

import os
import sys

test_bus_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(test_bus_base)

import pytest

from concurrency.bus import BusService, ServiceId
from pytests.unit.test_common import JobInt


@pytest.mark.parametrize('sid', [ServiceId.DETECTOR, ServiceId.SEGMENTATION])
def test_complete_queue(sid):
    BusService.get_queue_worker_to_app(sid).config_length(25)

    # add 4 numbers
    BusService.finish_job(sid, JobInt(1))
    BusService.finish_job(sid, JobInt(5))
    BusService.finish_job(sid, JobInt(9))
    BusService.finish_job(sid, JobInt(11))

    # got only 4
    assert BusService.fetch_complete_job(sid).number == 1
    assert BusService.fetch_complete_job(sid).number == 5
    assert BusService.fetch_complete_job(sid).number == 9
    assert BusService.fetch_complete_job(sid).number == 11
    assert BusService.fetch_complete_job(sid) is None


@pytest.mark.parametrize('sid', [ServiceId.DETECTOR, ServiceId.SEGMENTATION])
def test_complete_queue_overflow(sid):
    BusService.get_queue_worker_to_app(sid).config_length(3)  # overflow when inputs 4 numbers

    # add 4 numbers
    BusService.finish_job(sid, JobInt(1))
    BusService.finish_job(sid, JobInt(5))
    BusService.finish_job(sid, JobInt(9))
    BusService.finish_job(sid, JobInt(11))

    # got only 3
    assert BusService.fetch_complete_job(sid).number == 5
    assert BusService.fetch_complete_job(sid).number == 9
    assert BusService.fetch_complete_job(sid).number == 11
    assert BusService.fetch_complete_job(sid) is None


@pytest.mark.parametrize('sid', [ServiceId.DETECTOR, ServiceId.SEGMENTATION])
def test_pending_queue(sid):
    BusService.get_queue_app_to_worker(sid).config_length(25)

    # add 4 numbers
    BusService.send_job_to_worker(sid, JobInt(1))
    BusService.send_job_to_worker(sid, JobInt(5))
    BusService.send_job_to_worker(sid, JobInt(9))
    BusService.send_job_to_worker(sid, JobInt(11))

    # got only 4
    assert BusService.take_job(sid).number == 1
    assert BusService.take_job(sid).number == 5
    assert BusService.take_job(sid).number == 9
    assert BusService.take_job(sid).number == 11
    assert BusService.take_job(sid) is None


@pytest.mark.parametrize('sid', [ServiceId.DETECTOR, ServiceId.SEGMENTATION])
def test_pending_queue_overflow(sid):
    BusService.get_queue_app_to_worker(sid).config_length(3)

    # add 4 numbers
    BusService.send_job_to_worker(sid, JobInt(1))
    BusService.send_job_to_worker(sid, JobInt(5))
    BusService.send_job_to_worker(sid, JobInt(9))
    BusService.send_job_to_worker(sid, JobInt(11))

    # got only 4
    assert BusService.take_job(sid).number == 5
    assert BusService.take_job(sid).number == 9
    assert BusService.take_job(sid).number == 11
    assert BusService.take_job(sid) is None
