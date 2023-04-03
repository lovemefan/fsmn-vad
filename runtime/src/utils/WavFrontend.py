# -*- coding:utf-8 -*-
# @FileName  :WavFrontend.py
# @Time      :2023/4/3 17:10
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from typing import Tuple
import numpy as np

from runtime.src.utils.kaldifeat import compute_fbank_feats


class WavFrontend():
    """Conventional frontend structure for ASR.
    """

    def __init__(
            self,
            cmvn_file: str = None,
            fs: int = 16000,
            window: str = 'hamming',
            n_mels: int = 80,
            frame_length: int = 25,
            frame_shift: int = 10,
            filter_length_min: int = -1,
            filter_length_max: float = -1,
            lfr_m: int = 1,
            lfr_n: int = 1,
            dither: float = 1.0
    ) -> None:

        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither

    def forward_fbank(self,
                      input_content: np.ndarray,
                      ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_lens = [], []
        batch_size = input_content.shape[0]
        input_lengths = np.array([input_content.shape[1]])
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input_content[i][:waveform_length]
            waveform = waveform * (1 << 15)
            mat = compute_fbank_feats(waveform,
                                      num_mel_bins=self.n_mels,
                                      frame_length=self.frame_length,
                                      frame_shift=self.frame_shift,
                                      dither=self.dither,
                                      energy_floor=0.0,
                                      sample_frequency=self.fs)
            feats.append(mat)
            feats_lens.append(mat.shape[0])

        feats_pad = np.array(feats).astype(np.float32)
        feats_lens = np.array(feats_lens).astype(np.int64)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(self,
                         input_content: np.ndarray,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_lens = [], []
        batch_size = input_content.shape[0]

        if self.cmvn_file:
            cmvn = self.load_cmvn()

        input_lengths = np.array([input_content.shape[1]])
        for i in range(batch_size):
            mat = input_content[i, :input_lengths[i], :]

            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = self.apply_lfr(mat, self.lfr_m, self.lfr_n)

            if self.cmvn_file:
                mat = self.apply_cmvn(mat, cmvn)

            feats.append(mat)
            feats_lens.append(mat.shape[0])

        feats_pad = np.array(feats).astype(np.float32)
        feats_lens = np.array(feats_lens).astype(np.int32)
        return feats_pad, feats_lens

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append(
                    (inputs[i * lfr_n:i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n:].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray, cmvn: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs

    def load_cmvn(self,) -> np.ndarray:
        with open(self.cmvn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == '<AddShift>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    add_shift_line = line_item[3:(len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == '<Rescale>':
                line_item = lines[i + 1].split()
                if line_item[0] == '<LearnRateCoef>':
                    rescale_line = line_item[3:(len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue

        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn