# -*- coding:utf-8 -*-
# @FileName  :fsmnvad.py
# @Time      :2023/3/31 16:06
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

__author__ = "lovemefan"
__copyright__ = "Copyright (C) 2016 lovemefan"
__license__ = "MIT"
__version__ = "v0.0.1"

from runtime.src.utils.logger import setup_logger


class Vad(object):
    def __init__(self, mode=None, level='info'):
        setup_logger(level)

    def set_parameters(self, mode):
        pass

    def is_speech(self, buf, sample_rate, length=None):
        pass

    def segments(self):
        """get sements of audio"""
        pass


