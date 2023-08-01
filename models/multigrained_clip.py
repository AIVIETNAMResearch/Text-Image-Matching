from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import  build_CLIP_from_openai_pretrained, Transformer, LayerNorm

class CrossTransformerBlock(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        #self.cross_attn = nn.MultiheadAttention(embed_dim, embed_dim // 64, batch_first=True)
        self.cross_modal_transformer = Transformer(width=embed_dim, layers=1, heads=embed_dim // 64)

        # self.ln_pre_t = LayerNorm(embed_dim)
        # self.ln_pre_i = LayerNorm(embed_dim)
        # self.ln_post = LayerNorm(embed_dim)

        scale = self.cross_modal_transformer.width**-0.5

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        #nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        #nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def forward(self, x):
        x = self.cross_modal_transformer(x)
        return x
    
class MultiGrainedCLIP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_model, _ = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)

        #self.cross_transformer_blocks = nn.Sequential(*[CrossTransformerBlock(cfg.MODEL.EMBED_DIM) for _ in range(cfg.MODEL.CROSS_DEPTH)])
        self.cross_transformer = Transformer(width=cfg.MODEL.EMBED_DIM*2, layers=cfg.MODEL.CROSS_DEPTH, heads=cfg.MODEL.EMBED_DIM // 32)

        self.classifier = nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.NUM_CLASS)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)

    def encode_text(self, captions, return_dense=False):
        return self.base_model.encode_text(captions, return_dense=return_dense)

    def encode_image(self, images, return_dense=False):
        return self.base_model.encode_image(images, return_dense=return_dense)
        

    def forward(self, image, caption):
        image_features = self.base_model.encode_image(image, return_dense=False)
        text_features = self.base_model.encode_text(caption, return_dense=False)
        logit_scale = self.base_model.logit_scale
 
        text_image_features = torch.cat([text_features, image_features], dim=1)
        cross_features = self.cross_transformer(text_image_features)

        #image_features = image_features[:,0,:].float()
        #cross_image_features = cross_image_features[:, 0, :].float()
        #cross_image_features += image_features
        cross_image_features, cross_text_features = torch.split(cross_features, text_features.shape[1], dim=1)

        #text_features = text_features[torch.arange(text_features.shape[0]), caption.argmax(dim=-1)].float()
        #cross_text_features = cross_text_features[torch.arange(text_features.shape[0]), caption.argmax(dim=-1)].float()
        #cross_text_features += text_features        

        image_logits = self.classifier(text_features).float()
        text_logits = self.classifier(image_features).float()

        cross_image_logits = self.classifier(cross_text_features).float()
        cross_text_logits = self.classifier(cross_image_features).float()

        return {
            'global_features': (image_features, text_features),
            'cross_features': (cross_image_features, cross_text_features),
            'cross_logits': (cross_image_logits, cross_text_logits),
            'global_logits': (image_logits, text_logits),
            'logit_scale': logit_scale,
        }
