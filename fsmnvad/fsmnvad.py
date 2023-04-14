# -*- coding:utf-8 -*-
# @FileName  :fsmnvad.py
# @Time      :2023/3/31 16:06
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

__author__ = "lovemefan"
__copyright__ = "Copyright (C) 2016 lovemefan"
__license__ = "MIT"
__version__ = "v0.0.1"

import logging
import os.path
from pathlib import Path
from typing import Union

from fsmnvad.runtime.src.fsmnvad.Speech2VadSegmentOffline import E2EVadModelOffline
from fsmnvad.runtime.src.utils.AudioHelper import AudioReader
from fsmnvad.runtime.src.utils.logger import setup_logger
from fsmnvad.runtime.src.utils.tools import read_yaml
from fsmnvad.runtime.src.utils.WavFrontend import WavFrontend

root_dir = Path(os.path.dirname(os.path.abspath(__file__)))


class FSMNVad(object):
    def __init__(self, config_path=root_dir / "config/config.yaml", level="info"):
        self.config = read_yaml(config_path)
        self.frontend = WavFrontend(
            cmvn_file=root_dir / self.config["WavFrontend"]["cmvn_file"],
            **self.config["WavFrontend"]["frontend_conf"],
        )
        self.vad = E2EVadModelOffline(
            self.config["FSMN"], self.config["vadPostArgs"], root_dir
        )
        setup_logger(level)

    def set_parameters(self, mode):
        pass

    def extract_feature(self, waveform):
        fbank, _ = self.frontend.fbank(waveform)
        feats, feats_len = self.frontend.lfr_cmvn(fbank)
        return feats, feats_len

    def is_speech(self, buf, sample_rate=16000):
        assert sample_rate == 16000, "only support 16k sample rate"

    def segments_offline(self, waveform: Union[str, Path], sample_rate=16000):
        """get sements of audio"""
        if isinstance(waveform, Path):
            logging.info(f"load audio {waveform}")
            if not waveform.exists():
                raise FileExistsError(f"{waveform} is not exist.")
            if os.path.isfile(waveform):
                waveform, sample_rate = AudioReader.read_wav_file(waveform)
            else:
                raise FileNotFoundError(str(Path))
        assert (
            sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {sample_rate}"

        feats, feats_len = self.extract_feature(waveform)
        waveform = waveform[None, ...]
        segments_part, in_cache = self.vad.infer_offline(feats[None, ...], waveform, is_final=True)
        return segments_part[0]

    def segments_online(self):
        pass
