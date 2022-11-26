import os
import torch

from TTS.tts.utils.managers import EmbeddingManager

use_cuda = torch.cuda.is_available()

model_path = "models/model_se.pth.tar"
config_path = "models/config_se.config_se.json"

encoder_manager = EmbeddingManager(
    encoder_model_path=model_path,
    encoder_config_path=config_path,
    use_cuda=use_cuda,
)

audio_file = "test.aac"
embedd = encoder_manager.compute_embedding_from_clip(audio_file)
print(embedd)