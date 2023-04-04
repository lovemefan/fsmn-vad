

<br/>
<h2 align="center">FSMN VAD</h2>
<br/>

  A enterprise-grade [Voice Activity Detector](https://en.wikipedia.org/wiki/Voice_activity_detection) from [modelscope](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) and [funasr](https://github.com/alibaba-damo-academy/FunASR/).



<br/>
<h2 align="center">Key Features</h2>
<br/>

- **Fast**

  One audio (70s) less than **0.6s** to be processed on mac M1 pro . Under the ONNX runtime, the RTF can be accelerated to **129**

- **Lightweight**

  Do not need to download the model, the model is loaded from the memory directly.
  and the onnx model size is only **1.6M**.
  Do not need pytorch, torchaudio, etc. dependencies.

- **General**
  
  fsmn VAD was trained on chinese corpora. and it finished anti-noise training, with certain noise rejection ability performs well on audios from different domains with various background noise and quality levels.
  - [x] file vad
  - [ ] streaming vad
- **Flexible sampling rate**
  
  - [x] 16k
  - [ ] 8k
 
- **Highly Accurate**

  coming ... 
  
- **Highly Portable**

  fsmn VAD reaps benefits from the rich ecosystems built around **ONNX** running everywhere where these runtimes are available.



## Installation

```bash
git clone https://github.com/lovemefan/fsmn-vad
cd fsmn-vad
python setup.py install
```

## Usage

```python
from fsmnvad import FSMNVad
from pathlib import Path
vad = FSMNVad()
segments = vad.segments_offline(Path("/path/audio/vad_example.wav"))
print(segments)
```

## Citation
```
@inproceedings{zhang2018deep,
  title={Deep-FSMN for large vocabulary continuous speech recognition},
  author={Zhang, Shiliang and Lei, Ming and Yan, Zhijie and Dai, Lirong},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5869--5873},
  year={2018},
  organization={IEEE}
}
```
```
@misc{FunASR,
  author = {Speech Lab, Alibaba Group, China},
  title = {FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alibaba-damo-academy/FunASR/}},
}

```
