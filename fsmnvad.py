# -*- coding:utf-8 -*-
# @FileName  :fsmnvad.py
# @Time      :2023/3/31 16:06
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

__author__ = "lovemefan"
__copyright__ = "Copyright (C) 2016 lovemefan"
__license__ = "MIT"
__version__ = "v0.0.1"

import os.path
import time
from os import PathLike
from pathlib import Path
from typing import Union

from runtime.src.fsmnvad.E2EVadModel import E2EVadModel
from runtime.src.utils.AudioHelper import AudioReader
from runtime.src.utils.logger import setup_logger
from runtime.src.utils.tools import read_yaml
from runtime.src.utils.WavFrontend import WavFrontend


class FSMNVad(object):
    def __init__(self, config_path, level="info"):
        self.config = read_yaml(config_path)
        self.frontend = WavFrontend(
            cmvn_file=self.config["WavFrontend"]["cmvn_file"],
            **self.config["WavFrontend"]["frontend_conf"],
        )
        self.vad = E2EVadModel(self.config["FSMN"], self.config["vadPostArgs"])
        setup_logger(level)

    def set_parameters(self, mode):
        pass

    def extract_feature(self, waveform):
        fbank, _ = self.frontend.forward_fbank(waveform)
        feats, feats_len = self.frontend.forward_lfr_cmvn(fbank)
        return feats, feats_len

    def is_speech(self, buf, sample_rate=16000):
        assert sample_rate == 16000, "only support 16k sample rate"

    def segments_offline(self, waveform: Union[str, PathLike], sample_rate=16000):
        """get sements of audio"""
        if isinstance(waveform, PathLike):
            if os.path.isfile(waveform):
                waveform, sample_rate = AudioReader.read_wav_file(waveform)

        assert (
            sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {sample_rate}"

        waveform = waveform[None, ...]
        feats, feats_len = self.extract_feature(waveform)
        segments_part, in_cache = self.vad.infer_offline(feats, waveform, is_final=True)
        return segments_part[0]


if __name__ == "__main__":
    vad = FSMNVad("/Users/cenglingfan/Code/python-project/fsmn-vad/config/config.yaml")
    start = time.time()
    result = vad.segments_offline(Path("/Users/cenglingfan/Downloads/vad_example.wav"))
    end = time.time()
    print(end - start)
    print(result)
