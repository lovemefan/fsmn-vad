# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2023/4/4 17:00
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import fsmnvad
from pathlib import Path
from fsmnvad import AudioReader

vad = fsmnvad.FSMNVad()
segments = vad.segments_offline(Path("test/vad_example.wav"))
print(segments)

# online
in_cache = []
speech, sample_rate = AudioReader.read_wav_file('test/vad_example.wav')
speech_length = speech.shape[0]

sample_offset = 0
step = 1600
vad_online = fsmnvad.FSMNVadOnline()

for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
    if sample_offset + step >= speech_length - 1:
        step = speech_length - sample_offset
        is_final = True
    else:
        is_final = False
    segments_result, in_cache = vad_online.segments_online(
        speech[sample_offset: sample_offset + step],
        in_cache=in_cache, is_final=is_final)
    if segments_result:
        print(segments_result)