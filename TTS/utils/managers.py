import json
import random
from typing import Any, Dict, List, Tuple, Union

import fsspec
import numpy as np
import torch

from TTS.config import load_config
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.utils.audio import AudioProcessor


class EmbeddingManager():

    def __init__(self,encoder_model_path: str = "",encoder_config_path: str = "",use_cuda: bool = False):
        self.encoder = None
        self.encoder_ap = None
        self.use_cuda = use_cuda

        if encoder_model_path and encoder_config_path:
            self.init_encoder(encoder_model_path, encoder_config_path, use_cuda)

  
    def init_encoder(self, model_path: str, config_path: str, use_cuda=False) -> None:
      
        self.use_cuda = use_cuda
        self.encoder_config = load_config(config_path)
        self.encoder = setup_encoder_model(self.encoder_config)
        self.encoder_criterion = self.encoder.load_checkpoint(
            self.encoder_config, model_path, eval=True, use_cuda=use_cuda
        )
        self.encoder_ap = AudioProcessor(**self.encoder_config.audio)

    def compute_embedding_from_clip(self, wav_file: Union[str, List[str]]) -> list:
       
        def _compute(wav_file: str):
            waveform = self.encoder_ap.load_wav(wav_file, sr=self.encoder_ap.sample_rate)
            if not self.encoder_config.model_params.get("use_torch_spec", False):
                m_input = self.encoder_ap.melspectrogram(waveform)
                m_input = torch.from_numpy(m_input)
            else:
                m_input = torch.from_numpy(waveform)

            if self.use_cuda:
                m_input = m_input.cuda()
            m_input = m_input.unsqueeze(0)
            embedding = self.encoder.compute_embedding(m_input)
            return embedding

        if isinstance(wav_file, list):
            
            embeddings = None
            for wf in wav_file:
                embedding = _compute(wf)
                if embeddings is None:
                    embeddings = embedding
                else:
                    embeddings += embedding
            return (embeddings / len(wav_file))[0].tolist()
        embedding = _compute(wav_file)
        return embedding[0].tolist()