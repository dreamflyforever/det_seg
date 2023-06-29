# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:27
# @Author: YANG.C
# @File: test_instance_name.py

import sys
import os

test_instance_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(test_instance_base)

from concurrency.instance_name import InstanceName

if __name__ == '__main__':
    instance_rgbd = InstanceName(brief_name='test_rgbd')
    instance_rgbd.set_prefix('rgbd')
    instance_rgbd.set_postfix('$')
    print(instance_rgbd.m_fullname)  # rgbd/test_rgbd#$

    instance_rgbd.set_prefix_post_num('rgbd', 5)
    print(instance_rgbd.m_fullname)  # rgbd/test_rgbd#5

    instace_rgb = InstanceName(brief_name='test_rgb')
    instace_rgb.set_prefix('rgb')
    instace_rgb.set_postfix('$')
    print(instace_rgb.m_fullname)
