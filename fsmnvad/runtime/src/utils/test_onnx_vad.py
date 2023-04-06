# -*- coding:utf-8 -*-
# @FileName  :test_onnx.py
# @Time      :2023/4/3 18:53
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import numpy as np
import onnxruntime

if __name__ == "__main__":
    onnx_path = "/runtime/onnx/vad/onnx/fsmnvad-online.onnx"
    sess = onnxruntime.InferenceSession(onnx_path)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]

    def _get_feed_dict(feats_length):
        return {
            "speech": np.random.rand(1, feats_length, 400).astype(np.float32),
            "in_cache0": np.random.rand(1, 128, 19, 1).astype(np.float32),
            "in_cache1": np.random.rand(1, 128, 19, 1).astype(np.float32),
            "in_cache2": np.random.rand(1, 128, 19, 1).astype(np.float32),
            "in_cache3": np.random.rand(1, 128, 19, 1).astype(np.float32),
        }

    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        for name, value in zip(output_name, output):
            print("{}: {}".format(name, value.shape))

    _run(_get_feed_dict(100))
    _run(_get_feed_dict(200))
