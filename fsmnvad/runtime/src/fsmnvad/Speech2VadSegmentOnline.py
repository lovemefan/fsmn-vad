# -*- coding:utf-8 -*-
# @FileName  :Speech2VadSegmentOnline.py
# @Time      :2023/4/6 17:40
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from fsmnvad.runtime.src.fsmnvad.Speech2VadSegmentOffline import E2EVadModelOffline


class E2EVadModelOnline(E2EVadModelOffline):
    """Speech2VadSegmentOnline class
    """
    def __init__(self, **kwargs):
        super(E2EVadModelOffline, self).__init__(**kwargs)
        vad_cmvn_file = kwargs.get('vad_cmvn_file', None)
        self.frontend = None
        if self.vad_infer_args.frontend is not None:
            self.frontend = WavFrontend(cmvn_file=vad_cmvn_file, **self.vad_infer_args.frontend_conf)



    def infer_online(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            in_cache: Dict[str, torch.Tensor] = dict(), is_final: bool = False, max_end_sil: int = 800
    ) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        batch_size = speech.shape[0]
        segments = [[]] * batch_size
        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths, is_final)
            fbanks, _ = self.frontend.get_fbank()
        else:
            raise Exception("Need to extract feats first, please configure frontend configuration")
        if feats.shape[0]:
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            waveforms = self.frontend.get_waveforms()

            batch = {
                "feats": feats,
                "waveform": waveforms,
                "in_cache": in_cache,
                "is_final": is_final,
                "max_end_sil": max_end_sil
            }
            # a. To device
            batch = to_device(batch, device=self.device)
            segments, in_cache = self.vad_model.forward_online(**batch)
            # in_cache.update(batch['in_cache'])
            # in_cache = {key: value for key, value in batch['in_cache'].items()}
        return fbanks, segments, in_cache


def inference(
        batch_size: int,
        ngpu: int,
        log_level: Union[int, str],
        data_path_and_name_and_type,
        vad_infer_config: Optional[str],
        vad_model_file: Optional[str],
        vad_cmvn_file: Optional[str] = None,
        raw_inputs: Union[np.ndarray, torch.Tensor] = None,
        key_file: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        num_workers: int = 1,
        **kwargs,
):
    inference_pipeline = inference_modelscope(
        batch_size=batch_size,
        ngpu=ngpu,
        log_level=log_level,
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        key_file=key_file,
        allow_variable_data_keys=allow_variable_data_keys,
        output_dir=output_dir,
        dtype=dtype,
        seed=seed,
        num_workers=num_workers,
        **kwargs,
    )
    return inference_pipeline(data_path_and_name_and_type, raw_inputs)


def inference_modelscope(
        batch_size: int,
        ngpu: int,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        vad_infer_config: Optional[str],
        vad_model_file: Optional[str],
        vad_cmvn_file: Optional[str] = None,
        # raw_inputs: Union[np.ndarray, torch.Tensor] = None,
        key_file: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        num_workers: int = 1,
        **kwargs,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2vadsegment
    speech2vadsegment_kwargs = dict(
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        device=device,
        dtype=dtype,
    )
    logging.info("speech2vadsegment_kwargs: {}".format(speech2vadsegment_kwargs))
    speech2vadsegment = Speech2VadSegmentOnline(**speech2vadsegment_kwargs)

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            fs: dict = None,
            param_dict: dict = None,
    ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = VADTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=VADTask.build_preprocess_fn(speech2vadsegment.vad_infer_args, False),
            collate_fn=VADTask.build_collate_fn(speech2vadsegment.vad_infer_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
            ibest_writer = writer[f"1best_recog"]
        else:
            writer = None
            ibest_writer = None

        vad_results = []
        batch_in_cache = param_dict['in_cache'] if param_dict is not None else dict()
        is_final = param_dict.get('is_final', False) if param_dict is not None else False
        max_end_sil = param_dict.get('max_end_sil', 800) if param_dict is not None else 800
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch['in_cache'] = batch_in_cache
            batch['is_final'] = is_final
            batch['max_end_sil'] = max_end_sil

            # do vad segment
            _, results, param_dict['in_cache'] = speech2vadsegment(**batch)
            # param_dict['in_cache'] = batch['in_cache']
            if results:
                for i, _ in enumerate(keys):
                    if results[i]:
                        if "MODELSCOPE_ENVIRONMENT" in os.environ and os.environ["MODELSCOPE_ENVIRONMENT"] == "eas":
                            results[i] = json.dumps(results[i])
                        item = {'key': keys[i], 'value': results[i]}
                        vad_results.append(item)
                        if writer is not None:
                            results[i] = json.loads(results[i])
                            ibest_writer["text"][keys[i]] = "{}".format(results[i])

        return vad_results

    return _forward
