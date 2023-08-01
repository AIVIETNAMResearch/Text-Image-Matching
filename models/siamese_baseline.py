import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34
from .senet import se_resnext50_32x4d
from .efficientnet import EfficientNet
from transformers import AutoModel
import timm


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        # self.init_weight()
    def init_weight(self):
        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.q_proj.weight, std=0)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class SiameseBaselineModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__() 
        print(f"===========> Using Architecture: {self._get_name()}")
        self.model_cfg = model_cfg
        embed_dim = model_cfg.EMBED_DIM

        # visual branch of siamese network
        if self.model_cfg.IMG_ENCODER == "ViT":
            self.vis_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=768)
            self.img_in_dim = 768  
            self.domain_vis_fc = nn.Linear(self.img_in_dim, embed_dim)

        elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
            self.vis_backbone = se_resnext50_32x4d()
            self.img_in_dim = 2048
            self.domain_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)

        elif "efficientnet" in self.model_cfg.IMG_ENCODER:
            self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
            self.img_in_dim = self.vis_backbone.out_channels
            self.domain_vis_fc = nn.Linear(self.img_in_dim, embed_dim)

        
        # text branch of siamese network
        self.text_backbone = AutoModel.from_pretrained(self.model_cfg.TEXT_ENCODER)
        text_out_dim = self.text_backbone.config.hidden_size
        self.domian_lang_fc = nn.Sequential(
            nn.LayerNorm(text_out_dim),
            nn.Linear(text_out_dim, text_out_dim), 
            nn.ReLU(), 
            nn.Linear(text_out_dim, embed_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)


    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        return lang_embeds

    def encode_image(self, images):
        vis_embeds = self.vis_backbone(images)
        vis_embeds = self.domain_vis_fc(vis_embeds)
        return vis_embeds
    
    def forward(self, images, input_ids, attention_mask):
        vis_embeds = self.encode_image(images)
        lang_embeds = self.encode_text(input_ids, attention_mask)
        return vis_embeds, lang_embeds, self.logit_scale

