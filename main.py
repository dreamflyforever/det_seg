# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:00
# @Author: YANG.C
# @File: main.py


import sys
import os

sys.path.append('../pyrobot')

from configs.config import cfg
from utils.log import logger, set_logger


def main():
    log_file = cfg['log_file']
    set_logger(logger, log_file)
    logger.debug('init logger system successful')
    logger.debug(cfg)


if __name__ == '__main__':
    main()
