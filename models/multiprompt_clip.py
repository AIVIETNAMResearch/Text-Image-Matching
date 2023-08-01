from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import  build_CLIP_from_openai_pretrained
from .multi_clip import prediction_MLP, projection_MLP


class MultiPromptCLIP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_model, _ = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)

    def encode_image(self, images: torch.Tensor, return_dense=False) -> torch.Tensor:
        return self.base_model.encode_image(images, return_dense=return_dense)

    def encode_text(self, captions: torch.Tensor, return_dense=False) -> torch.Tensor:
        return self.base_model.encode_text(captions, return_dense=return_dense)    
    
    def forward(self, images, captions, noun_chunks):
        image_features = self.encode_image(images, return_dense=True)
        text_features = self.encode_text(captions, return_dense=True)

        global_image_features = image_features[:, 0, :]
        global_text_features = text_features[torch.arange(text_features.shape[0]), captions.argmax(dim=-1)]
        repeat_cap = text_features.repeat(1, 6, 1).reshape(images.shape[0], 6, 77, 512)
        mask = torch.zeros((images.shape[0], 6, 77, 512))

        for batch in range(images.shape[0]):
            #print(new_interest_tokens[batch])
            for i, temp_index in enumerate(noun_chunks[batch]):
                mask[batch, i, temp_index] = 1
        mask = mask.to(images.device)
        local_text_features = repeat_cap * mask
        local_text_features = local_text_features.mean(dim=2)

        return {
            "global_features": (global_text_features, global_image_features),
            "local_text_features": local_text_features,
            "logit_scale": self.base_model.logit_scale
        }