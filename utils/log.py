# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:58
# @Author: YANG.C
# @File: logger.py

# import logging
#
# # 定义自定义的日志输出格式
# LOG_COLORS = {
#     logging.DEBUG: "\033[0;37m",  # 灰色
#     logging.INFO: "\033[0;36m",  # 青色
#     logging.WARNING: "\033[0;33m",  # 黄色
#     logging.ERROR: "\033[0;31m",  # 红色
#     logging.CRITICAL: "\033[0;35m"  # 紫色
# }
#
#
# class ColoredFormatter(logging.Formatter):
#     def format(self, record):
#         level_color = LOG_COLORS.get(record.levelno)
#         msg = super().format(record)
#         if level_color:
#             msg = level_color + msg + "\033[0m"  # 给msg添加颜色代码
#         return msg
#
#
# format_str = "%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
# format = logging.Formatter(format_str)
# logging.basicConfig(format=format_str,
#                     datefmt='%Y-%m-%d %H:%M:%S')
#
# logging.getLogger().setLevel(logging.DEBUG)
# logger = logging.getLogger()

import logging

# 定义自定义的日志输出格式
LOG_COLORS = {
    logging.DEBUG: "\033[0;37m",  # 灰色
    logging.INFO: "\033[0;36m",  # 青色
    logging.WARNING: "\033[0;33m",  # 黄色
    logging.ERROR: "\033[0;31m",  # 红色
    logging.CRITICAL: "\033[0;35m"  # 紫色
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level_color = LOG_COLORS.get(record.levelno)
        msg = super().format(record)
        if level_color:
            msg = level_color + msg + "\033[0m"  # 给msg添加颜色代码
        return msg


# 设置全局日志级别和根记录器的级别
logging.getLogger().setLevel(logging.DEBUG)

# 获取根记录器，并清空其已有的处理器和过滤器
logger = logging.getLogger()
logger.handlers = []
logger.filters = []

# 创建一个StreamHandler，用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_formatter = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")

# 设置StreamHandler使用自定义Formatter
console_formatter = ColoredFormatter("%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
console_handler.setFormatter(console_formatter)

# 将StreamHandler添加到logger对象中
logger.addHandler(console_handler)


def set_logger(logger, log_file):
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
