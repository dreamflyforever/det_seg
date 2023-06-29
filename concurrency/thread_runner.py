# -*- coding: utf-8 -*-
# @Time: 2023/6/27 上午11:46
# @Author: YANG.C
# @File: thread_runner.py

from __future__ import annotations

import os
import sys

queue_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(queue_base)

import multiprocessing
import queue
import threading
import time
from abc import ABC, abstractmethod
from enum import IntEnum

from concurrency.instance_name import InstanceName
from utils.log import logger


class ThreadMode(IntEnum):
    InProcess = 1
    Threaded = 2
    Async = 4
    AllModes = 7  # 7 = 4 + 2 + 1


class ThreadState(IntEnum):
    Fresh = 1
    Prepared = 3
    Running = 4
    Stopped = 5
    Error = 6


class ThreadRunner(InstanceName, ABC):
    def __init__(self):
        super().__init__()
        self.m_thread: multiprocessing.Process | threading.Thread | None = None

        # sleeping time between interations or calling of pump() in Thread/InProcess mode
        self.m_breathing_ms = 1
        self.m_queueToWorker: multiprocessing.Queue | queue.Queue | None = None
        self.m_queueFromWorker: multiprocessing.Queue | queue.Queue | None = None
        self.m_shared = multiprocessing.Manager().Namespace()
        self.m_shared.stop_flag = False
        self.m_shared.state = ThreadState.Fresh
        self.m_shared.running_mode = ThreadMode.InProcess

    def clear(self) -> None:
        # if this assert is triggered it is highly possible that stop_and_wait() is not called before.
        assert (not self.is_running()) and (not self.is_prepared())

        # when ThreadRunner.run() exits, the thread may have not joined.
        if self.m_thread is not None:
            self.m_thread.join()
            self.m_thread = None
            logger.info(f'{self.fullname()}, Thread / Process for Runner destructed!')

    def get_in_out_queue(self) -> tuple[multiprocessing.Queue, multiprocessing.Queue]:
        return self.m_queueToWorker, self.m_queueFromWorker

    def get_state(self) -> ThreadState:
        return self.m_shared.state

    def get_running_mode(self) -> ThreadMode:
        return self.m_shared.running_mode

    def is_supported(self, mode: ThreadMode) -> bool:
        return self.get_supported_mode() & mode == mode

    @staticmethod
    def ms_sleep(milliseconds) -> None:
        time.sleep(milliseconds / 1000.)
        return

    @staticmethod
    def thread_entry_func(runner: ThreadRunner) -> None:
        runner.thread_func_for_run()

    def need_to_stop(self) -> bool:
        return self.m_shared.stop_flag

    def set_running_model(self, mode: ThreadMode) -> None:
        assert self.is_fresh()
        assert self.is_supported(mode)
        assert mode != ThreadMode.AllModes
        self.m_shared.running_mode = mode

    # must be called frequently in async mode
    def pump(self) -> None:
        if not self.is_running():
            return
        if self.is_supported(ThreadMode.Async) and self.get_running_mode() == ThreadMode.Async:
            self._run_pump()
        else:
            pass  # do nothing for InProcess / Threaded mode

    def get_supported_mode(self) -> ThreadMode:
        return ThreadMode.InProcess

    @abstractmethod
    def _run_pre(self) -> None:
        raise AssertionError("Not implemented")

    @abstractmethod
    def _run_pump(self) -> None:
        raise AssertionError("Not implemented")

    @abstractmethod
    def _run_post(self) -> None:
        raise AssertionError("Not implemented")

    def start_run(self) -> None:
        assert self.is_fresh()
        self.prepare_blocked()
        if not self.is_prepared():
            return

        running_mode = self.get_running_mode()
        if running_mode == ThreadMode.Async:
            logger.info(f'{self.fullname()}, ++++++ start a new async-runner ++++++')
            self.m_queueFromWorker = queue.Queue()
            self.m_queueToWorker = queue.Queue()
            self.m_shared.state = ThreadState.Running
            self._run_pre()
            return

            # goes here if the running mode is either InProcess or Threaded
        assert self.m_thread is None
        self.m_shared.state = ThreadState.Running

        # create the Thread / InProcess
        if self.get_running_mode() == ThreadMode.InProcess:
            logger.info(f'{self.fullname()}, ++++++ start a new Process ++++++')
            self.m_queueFromWorker = multiprocessing.Queue()
            self.m_queueToWorker = multiprocessing.Queue()
            self.m_thread = multiprocessing.Process(target=ThreadRunner.thread_entry_func,
                                                    args=(self,))
        else:
            logger.info(f'{self.fullname()}, ++++++ start a new Thread ++++++')
            self.m_queueFromWorker = queue.Queue()
            self.m_queueToWorker = queue.Queue()
            self.m_thread = threading.Thread(target=ThreadRunner.thread_entry_func, args=(self,))

        # fork the crated Thread / InProcess
        self.m_thread.start()

    def prepare_blocked(self) -> None:
        assert ThreadState.Fresh == self.get_state()
        logger.info(f'{self.fullname()}, ======= Prepare in blocked mode ======')
        if self._prepare_to_run():
            self.m_shared.state = ThreadState.Prepared
            logger.info(f'{self.fullname()}, Preparation finished.')
        else:
            self.m_shared.state = ThreadState.Error
            logger.info(f'{self.fullname()}, Preparation failed.')

    # call this method to safely stop and join the Thread / InProcess
    def stop_and_wait(self) -> None:
        # caution to call virual functions here not
        logger.info(f'{self.fullname()}, Stop Thread/InProcess and wait...')
        self.stop()
        if self.m_thread is not None:
            self.m_thread.join()
            self.m_thread = None
        logger.info(f'{self.fullname()}, End of waiting for stop.')

    def is_fresh(self) -> bool:
        return ThreadState.Fresh == self.get_state()

    def is_prepared(self) -> bool:
        return ThreadState.Prepared == self.get_state()

    def is_running(self) -> bool:
        return ThreadState.Running == self.get_state()

    def is_error(self) -> bool:
        return ThreadState.Error == self.get_state()

    # call this method to request the thread / process to stop.
    # It returns immediately and doesn't wait for the thread / process to fully stop.
    def stop(self) -> None:
        logger.info(f'{self.fullname()}, Stopping thread...')
        self.m_shared.stop_flag = True
        if self.get_running_mode() == ThreadMode.Async:
            self._run_post()
            self.m_shared.state = ThreadState.Stopped
            self.m_shared.stop_flag = False
            logger.info(f'{self.fullname()}, Exits async runner...')

    def thread_func_for_run(self) -> None:
        assert self.is_supported(ThreadMode.Threaded) or self.is_supported(ThreadMode.InProcess)
        assert self.get_running_mode() * self.get_supported_mode() > 0
        logger.info(f'{self.fullname()}, ****** Thread is running ******')

        # start of threaded task
        self._run_pre()
        while not self.need_to_stop():
            # will be overrided.
            to_breath = self._run_pump()
            if self.m_breathing_ms > 0 and to_breath and self.get_running_mode() != ThreadMode.Async:
                # sleep only for Threaded and InProcess mode
                self.ms_sleep(self.m_breathing_ms)
        self._run_post()
        # end of threaded task

        self.m_shared.state = ThreadState.Stopped
        self.m_shared.stop_flag = False
        logger.info(f'{self.fullname()}, ----- Exits Thread ------.')

    # Note:
    # Override this method to add preparation codes.
    # It's different from _run_pre(): _run_pre is called in separate Thread / Process, but
    # _prepare_to_run is running in caller's thread.
    def _prepare_to_run(self) -> bool:
        return True

    # convenient function for preparing a list of runners.
    @staticmethod
    def prepare_multiple(runners: list[ThreadRunner]) -> bool:
        # prepare all runners
        for runner in runners:
            if runner is not None:
                runner.prepare_blocked()

        # check if all runners are prepared
        all_prepared = False
        for runner in runners:
            if runner is not None:
                if not runner.is_prepared():
                    logger.error(f'Thread [{runner.fullname}] is not prepared!')
                    all_prepared = False
                    break

        return all_prepared

    # convenient function for running a list of runners
    @staticmethod
    def start_run_multiple(runners: list[ThreadRunner]) -> bool:
        # prepare all runners
        ThreadRunner.prepare_multiple(runners)

        # start components
        for runner in runners:
            runner.start_run()

        # check if all workers are running
        all_running = True
        for runner in runners:
            while not runner.is_running():
                all_running = False
                logger.info(f'Waiting Thread [{runner.fullname()}] to run...')

        return all_running

    # convenient function for stopping a list of runners
    @staticmethod
    def stop_and_wait_multiple(runners: list[ThreadRunner], timeout_ms: int) -> int:
        # stop runners
        for runner in runners:
            runner.stop()

        # check if all workers are running
        all_stopped = False
        for runner in runners:
            while runner.is_running():
                all_stopped = False
                logger.info(f'Waiting Thread [{runner.fullname()}] to run...')

        return all_stopped

    # convenient function for retaining a list of runners
    @staticmethod
    def pump_multiple(runners: list[ThreadRunner]) -> None:
        for runner in runners:
            runner.pump()
