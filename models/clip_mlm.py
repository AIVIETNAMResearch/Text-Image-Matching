from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import  build_CLIP_from_openai_pretrained, LayerNorm, Transformer, QuickGELU

class CLIPMLM(nn.Module):
    def __init__(self, cfg):
        super(CLIPMLM, self).__init__()
        self.base_model, _ = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)
        
        self.classifier = nn.Linear(512, cfg.MODEL.NUM_CLASS)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)

        self.cross_attn = nn.MultiheadAttention(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM // 64, batch_first=True)
        self.cross_modal_transformer = Transformer(width=cfg.MODEL.EMBED_DIM, layers=cfg.MODEL.CROSS_DEPTH, heads=cfg.MODEL.EMBED_DIM // 64)
        
        self.ln_pre_t = LayerNorm(cfg.MODEL.EMBED_DIM)
        self.ln_pre_i = LayerNorm(cfg.MODEL.EMBED_DIM)
        self.ln_post = LayerNorm(cfg.MODEL.EMBED_DIM)
        
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
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # mlm classifier
        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM)),
                    ('gelu', QuickGELU()),
                    ('ln', LayerNorm(cfg.MODEL.EMBED_DIM)),
                    ('fc', nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.VOCAB_SIZE))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def encode_image(self, images, return_dense=False):
        return self.base_model.encode_image(images, return_dense=return_dense)
    
    def encode_text(self, captions, return_dense=False):
        return self.base_model.encode_text(captions, return_dense=return_dense)

    def cross_transformer(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q), 
            self.ln_pre_i(k), 
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    

    def forward(self, images, captions, captions_mlm):
        image_feats = self.encode_image(images, return_dense=True)
        text_feats = self.encode_text(captions)
        

        mlm_feats = self.encode_text(captions_mlm, return_dense=True)
        x = self.cross_transformer(mlm_feats, image_feats, image_feats)
        mlm_scores = self.mlm_head(x)

        image_feats = image_feats[:, 0, :]
        # ID Classifer
        image_logits = self.classifier(image_feats)
        text_logits = self.classifier(text_feats)

        return {
            "image_feats": image_feats,
            "text_feats": text_feats,
            "logit_scale": self.base_model.logit_scale,
            "id_logits": [image_logits, text_logits],
            "mlm_scores": mlm_scores
        }

