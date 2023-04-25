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
from typing import Union, Tuple

import numpy as np

from fsmnvad.runtime.src.fsmnvad.Speech2VadSegmentOffline import E2EVadModel
from fsmnvad.runtime.src.utils.AudioHelper import AudioReader
from fsmnvad.runtime.src.utils.logger import setup_logger
from fsmnvad.runtime.src.utils.tools import read_yaml
from fsmnvad.runtime.src.utils.WavFrontend import WavFrontend, WavFrontendOnline

root_dir = Path(os.path.dirname(os.path.abspath(__file__)))


class FSMNVad(object):
    def __init__(self, config_path=root_dir / "config/config.yaml", level="info", online=False):
        self.config = read_yaml(config_path)

        if online:
            self.frontend = WavFrontendOnline(
                cmvn_file=root_dir / self.config["WavFrontend"]["cmvn_file"],
                **self.config["WavFrontend"]["frontend_conf"],
            )
            self.config["FSMN"]["model_path"] = self.config["FSMN"]["online_model_path"]
        else:
            self.frontend = WavFrontend(
                cmvn_file=root_dir / self.config["WavFrontend"]["cmvn_file"],
                **self.config["WavFrontend"]["frontend_conf"],
            )
            self.config["FSMN"]["model_path"] = self.config["FSMN"]["offline_model_path"]
        self.vad = E2EVadModel(
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

    def segments_offline(self, waveform_path: Union[str, Path]):
        """get sements of audio"""

        logging.info(f"load audio {waveform_path}")
        if not os.path.exists(waveform_path):
            raise FileExistsError(f"{waveform_path} is not exist.")
        if os.path.isfile(waveform_path):
            waveform, _sample_rate = AudioReader.read_wav_file(waveform_path)
        else:
            raise FileNotFoundError(str(Path))
        assert (
                _sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {_sample_rate}"

        feats, feats_len = self.extract_feature(waveform)
        waveform = waveform[None, ...]
        segments_part, in_cache = self.vad.infer_offline(feats[None, ...], waveform, is_final=True)
        return segments_part[0]


class FSMNVadOnline(FSMNVad):
    def __init__(self, config_path=root_dir / "config/config.yaml", level="info"):
        super(FSMNVadOnline, self).__init__(config_path, level, online=True)


    def set_parameters(self, mode):
        pass

    def extract_feature(self, waveform):
        waveform = waveform * (1 << 15)
        # feats, feats_len = self.frontend.extract_feat(waveform[None, ...])
        feats, feats_len = self.frontend.extract_fbank(waveform[None, ...])
        # feats, feats_len, _ = self.frontend.lfr_cmvn(fbank, [int(feats_len)])
        return feats.astype(np.float32), feats_len.astype(np.int32)

    def is_speech(self, buf, sample_rate=16000):
        assert sample_rate == 16000, "only support 16k sample rate"

    def prepare_cache(self, in_cache: list):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.config["FSMN"]["encoder_conf"]["fsmn_layers"]
        proj_dim = self.config["FSMN"]["encoder_conf"]["proj_dim"]
        lorder = self.config["FSMN"]["encoder_conf"]["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def segments_online(self, waveform: Union[str, np.ndarray],
                        sample_rate=16000,
                        in_cache=None,
                        is_final=False):
        """get sements of audio"""

        if in_cache is None:
            in_cache = []

        if isinstance(waveform, str):
                waveform = AudioReader.read_pcm_byte(waveform)

        assert (
                sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {sample_rate}"
        feats, feats_len = self.extract_feature(waveform)
        waveform = self.frontend.get_waveforms()
        segments_part, in_cache = self.vad.infer_online(feats,
                                                        waveform,
                                                        self.prepare_cache(in_cache),
                                                        is_final=is_final)
        return segments_part, in_cache
