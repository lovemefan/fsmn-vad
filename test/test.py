# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2023/4/4 17:00
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import fsmnvad
from pathlib import Path
vad = fsmnvad.FSMNVad()
segments = vad.segments_offline(Path("vad_example.wav"))
print(segments)