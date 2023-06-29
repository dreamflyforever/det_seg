# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:36
# @Author: YANG.C
# @File: test_log.py

import sys
import os
import logging

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base)

from utils.log import logger, file_formatter

if __name__ == '__main__':
    log_file = './log.txt'

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    logger.error('test_error')
    logger.warning('test_warning')
    logger.debug('test_debug')
    logger.info('test_info')
