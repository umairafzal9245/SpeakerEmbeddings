import os
import torch

from TTS.tts.utils.managers import EmbeddingManager

use_cuda = torch.cuda.is_available()

model_path = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
config_path = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
old_file = ""   # previous embeddings file 

encoder_manager = EmbeddingManager(
    encoder_model_path=model_path,
    encoder_config_path=config_path,
    use_cuda=use_cuda,
)

audio_file = "test.wav"
embedd = encoder_manager.compute_embedding_from_clip(audio_file)
print(embedd)