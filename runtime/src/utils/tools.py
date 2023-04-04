# -*- coding:utf-8 -*-
# @FileName  :tools.py
# @Time      :2023/4/3 19:17
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from pathlib import Path
from typing import Dict, Union

import yaml


def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
