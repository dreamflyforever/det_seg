# -*- coding: utf-8 -*-
# @Time: 2023/6/26 下午5:14
# @Author: YANG.C
# @File: instance_name.py

from __future__ import annotations


class InstanceName:
    def __init__(self, brief_name="No_name"):
        self.m_brief_name = brief_name
        self.m_prefix = ''
        self.m_postfix = ''
        self.m_fullname = '/No_name'
        self.m_prefix_delimiter = '/'
        self.m_num_delimiter = '#'
        self._update_fullname()

    def fullname(self) -> str:
        return self.m_fullname

    def set_prefix(self, prefix: str) -> None:
        assert len(prefix) > 0
        self.m_prefix = prefix
        self._update_fullname()

    def set_postfix(self, postfix: str) -> None:
        assert len(postfix) > 0
        self.m_postfix = postfix
        self._update_fullname()

    def set_post_num(self, num: int) -> None:
        self.set_postfix(str(num))

    def set_prefix_postfix(self, prefix, postfix: str) -> None:
        self.set_prefix(prefix)
        self.set_postfix(postfix)
        self._update_fullname()

    def set_prefix_post_num(self, prefix: str, num: int) -> None:
        self.set_prefix_postfix(prefix, str(num))

    def set_brief_name(self, name) -> str:
        assert len(str(name)) > 0
        self.m_brief_name = str(name)
        self._update_fullname()

    def get_prefix(self) -> str:
        return self.m_prefix

    def get_postfix(self) -> str:
        return self.m_postfix

    def get_brief_name(self) -> str:
        return self.m_brief_name

    def get_fullname(self) -> str:
        return self.m_fullname

    def _update_fullname(self) -> None:
        assert len(self.m_brief_name) > 0
        self.m_fullname = self.m_brief_name
        if len(self.m_prefix) > 0:
            self.m_fullname = self.m_prefix + self.m_prefix_delimiter + self.m_fullname
        if len(self.m_postfix) > 0:
            self.m_fullname = self.m_fullname + self.m_num_delimiter + self.m_postfix
