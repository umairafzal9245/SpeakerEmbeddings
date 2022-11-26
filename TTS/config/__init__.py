import json
import os
import re
from typing import Dict

import fsspec
import yaml
from coqpit import Coqpit

from TTS.config.shared_configs import *
from TTS.utils.generic_utils import find_module


def read_json_with_comments(json_path):
    
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
  
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data


def register_config(model_name: str) -> Coqpit:
    
    config_class = None
    config_name = model_name + "_config"
    paths = ["TTS.tts.configs", "TTS.vocoder.configs", "TTS.encoder.configs"]
    for path in paths:
        try:
            config_class = find_module(path, config_name)
        except ModuleNotFoundError:
            pass
    if config_class is None:
        raise ModuleNotFoundError(f" [!] Config for {model_name} cannot be found.")
    return config_class


def _process_model_name(config_dict: Dict) -> str:
   
    model_name = config_dict["model"] if "model" in config_dict else config_dict["generator_model"]
    model_name = model_name.replace("_generator", "").replace("_discriminator", "")
    return model_name


def load_config(config_path: str) -> Coqpit:
    
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config