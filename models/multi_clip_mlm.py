from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import  build_CLIP_from_openai_pretrained, LayerNorm, Transformer, QuickGELU

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024, num_layers=3):
        super(projection_MLP, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = BN(hidden_dim)

        if self.num_layers == 3:
            self.relu2 = nn.ReLU(inplace=True)
            self.linear3 = nn.Linear(hidden_dim, out_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            # self.bn3 = BN(hidden_dim)

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # b, _ = x.shape
        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        # layer 2
        x = self.linear2(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn2(x)


        if self.num_layers == 3:
            x = self.relu2(x)
            # x.reshape(b, self.hidden_dim)
            # layer 3
            x = self.linear3(x)
            # x.reshape(b, self.out_dim, 1)
            x = self.bn3(x)
            # x.reshape(b, self.out_dim)

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=1024): # bottleneck structure
        super(prediction_MLP, self).__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        b, _ = x.shape

        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        x = self.layer2(x)
        return x

class MultiViewCLIPMLM(nn.Module):
    def __init__(self, cfg):
        super(MultiViewCLIPMLM, self).__init__()
        self.base_model, _ = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)

        # SimSiam
        self.projector_image = projection_MLP(512, hidden_dim=512, out_dim=512)
        self.predictor_image = prediction_MLP(512, hidden_dim=512, out_dim=512)

        self.projector_text = projection_MLP(512, hidden_dim=512, out_dim=512)
        self.predictor_text = prediction_MLP(512, hidden_dim=512, out_dim=512)
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
    

    def forward(self, images1, images2, captions1, captions2, captions_mlm):
        image_feats = self.encode_image(torch.cat([images1, images2], dim=0), return_dense=True)
        text_feats = self.encode_text(torch.cat([captions1, captions2], dim=0), return_dense=True)
        
        ori_image_feats_1, ori_image_feats_2 = torch.split(image_feats, images1.shape[0], dim=0)
        ori_text_feats_1, ori_text_feats_2 = torch.split(text_feats, captions1.shape[0], dim=0)
        
        image_feats_1 = ori_image_feats_1[:, 0, :]
        image_feats_2 = ori_image_feats_2[:, 0, :]
        text_feats_1 = ori_text_feats_1[torch.arange(ori_text_feats_1.shape[0]), captions1.argmax(dim=-1)]
        text_feats_2 = ori_text_feats_2[torch.arange(ori_text_feats_2.shape[0]), captions2.argmax(dim=-1)]
        
        # SimSiam
        z1_image = self.projector_image(image_feats_1)
        z2_image = self.projector_image(image_feats_2)
        p1_image = self.predictor_image(z1_image)
        p2_image = self.predictor_image(z2_image)

        z1_text = self.projector_text(text_feats_1)
        z2_text = self.projector_text(text_feats_2)
        p1_text = self.predictor_text(z1_text)
        p2_text = self.predictor_text(z2_text)

        # ID Classifer
        image_logits_1 = self.classifier(image_feats_1)
        text_logits_1 = self.classifier(text_feats_1)
        image_logits_2 = self.classifier(image_feats_2)
        text_logits_2 = self.classifier(text_feats_2)

        mlm_feats = self.encode_text(captions_mlm, return_dense=True)
        x = self.cross_transformer(mlm_feats, ori_image_feats_1, ori_image_feats_1)
        mlm_scores = self.mlm_head(x)

        return {
            "image_feats": [image_feats_1, image_feats_2],
            "text_feats": [text_feats_1, text_feats_2],
            "simsiam_features_images": [z1_image, z2_image, p1_image, p2_image],
            "simsiam_features_texts": [z1_text, z2_text, p1_text, p2_text],
            "logit_scale": self.base_model.logit_scale,
            "id_logits": [image_logits_1, text_logits_1, image_logits_2, text_logits_2],
            "mlm_scores": mlm_scores
        }


class MAEDecoder(nn.Module):
    def __init__(self, cfg, base_model):
        super(MAEDecoder, self).__init__()
        self.cfg = cfg
        patch_size = cfg.MODEL.STRIDE
        self.patch_size = patch_size
        self.num_x = base_model.num_x
        self.num_y = base_model.num_y

        num_patches = self.num_x * self.num_y
        self.base_model = base_model


        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.EMBED_DIM))

        self.cross_attn = nn.MultiheadAttention(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM // 64, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.MODEL.EMBED_DIM,
                out_channels=cfg.MODEL.STRIDE ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(cfg.MODEL.STRIDE),
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward(self, imgs, mask_ratio, text_embs=None):
        # embed patches
        #x = imgs.view(imgs.shape[0], imgs.shape[1] // self.num_x, imgs.shape[1] // self.num_y)
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)
        #x = self.base_model.encode_image(x)
        x = self.base_model.conv1(imgs)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) # [*, num_patches, embed_dim]
        x = x + self.base_model.positional_embedding.to(x.dtype)[1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.base_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        cls_token = cls_token + self.base_model.positional_embedding.to(x.dtype)[:1, :]

        x = torch.cat([cls_token, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = self.base_model.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.base_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.base_model.ln_post(x)
        if self.base_model.proj is not None:
            x = x @ self.base_model.proj
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])) #unshuffle
        #x = torch.cat([x[:, :1, :], x_], dim=1) # append cls token

        if text_embs != None:
            x = self.cross_attn(x,text_embs,text_embs)[0]
        
        pred = self.decoder(x.reshape(x.shape[0], x.shape[-1], self.num_y, self.num_x))
        
        pred= self.patchify(pred)
        imgs = self.patchify(imgs)
        return imgs, pred, mask




class MultiViewCLIPMLMMAE(nn.Module):
    def __init__(self, cfg):
        super(MultiViewCLIPMLMMAE, self).__init__()
        self.base_model, _ = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)

        # SimSiam
        self.projector_image = projection_MLP(512, hidden_dim=512, out_dim=512)
        self.predictor_image = prediction_MLP(512, hidden_dim=512, out_dim=512)

        self.projector_text = projection_MLP(512, hidden_dim=512, out_dim=512)
        self.predictor_text = prediction_MLP(512, hidden_dim=512, out_dim=512)
        
        self.classifier = nn.Linear(512, cfg.MODEL.NUM_CLASS)

        self.cross_attn = nn.MultiheadAttention(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM // 64, batch_first=True)
        self.cross_modal_transformer = Transformer(width=cfg.MODEL.EMBED_DIM, layers=cfg.MODEL.CROSS_DEPTH, heads=cfg.MODEL.EMBED_DIM // 64)
        
        self.ln_pre_t = LayerNorm(cfg.MODEL.EMBED_DIM)
        self.ln_pre_i = LayerNorm(cfg.MODEL.EMBED_DIM)
        self.ln_post = LayerNorm(cfg.MODEL.EMBED_DIM)
        
        scale = cfg.MODEL.EMBED_DIM**-0.5

        # mlm classifier
        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM)),
                    ('gelu', QuickGELU()),
                    ('ln', LayerNorm(cfg.MODEL.EMBED_DIM)),
                    ('fc', nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.VOCAB_SIZE))]))

        self.cls_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM)),
                    ('gelu', QuickGELU()),
                    ('ln', LayerNorm(cfg.MODEL.EMBED_DIM)),
                    ('fc', nn.Linear(cfg.MODEL.EMBED_DIM, 1))]))

        ########################
        # Decoder Model
        ########################
        self.decoder = MAEDecoder(cfg, self.base_model.visual)
        self.mask_ratio = cfg.MODEL.MASK_RATIO

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
    

    def forward(self, images1, images2, captions1, captions2, captions_mlm):
        image_feats = self.encode_image(torch.cat([images1, images2], dim=0), return_dense=True)
        text_feats = self.encode_text(torch.cat([captions1, captions2], dim=0), return_dense=True)
        
        ori_image_feats_1, ori_image_feats_2 = torch.split(image_feats, images1.shape[0], dim=0)
        ori_text_feats_1, ori_text_feats_2 = torch.split(text_feats, captions1.shape[0], dim=0)
        
        image_feats_1 = ori_image_feats_1[:, 0, :]
        image_feats_2 = ori_image_feats_2[:, 0, :]
        text_feats_1 = ori_text_feats_1[torch.arange(ori_text_feats_1.shape[0]), captions1.argmax(dim=-1)]
        text_feats_2 = ori_text_feats_2[torch.arange(ori_text_feats_2.shape[0]), captions2.argmax(dim=-1)]
        
        # SimSiam
        z1_image = self.projector_image(image_feats_1)
        z2_image = self.projector_image(image_feats_2)
        p1_image = self.predictor_image(z1_image)
        p2_image = self.predictor_image(z2_image)

        z1_text = self.projector_text(text_feats_1)
        z2_text = self.projector_text(text_feats_2)
        p1_text = self.predictor_text(z1_text)
        p2_text = self.predictor_text(z2_text)

        # ID Classifer
        image_logits_1 = self.classifier(image_feats_1)
        text_logits_1 = self.classifier(text_feats_1)
        image_logits_2 = self.classifier(image_feats_2)
        text_logits_2 = self.classifier(text_feats_2)

        mlm_feats = self.encode_text(captions_mlm, return_dense=True)
        x = self.cross_transformer(mlm_feats, ori_image_feats_1, ori_image_feats_1)
        mlm_scores = self.mlm_head(x)

        imgs, pred, mask = self.decoder(images1, self.mask_ratio, ori_text_feats_1)


        return {
            "image_feats": [image_feats_1, image_feats_2],
            "text_feats": [text_feats_1, text_feats_2],
            "simsiam_features_images": [z1_image, z2_image, p1_image, p2_image],
            "simsiam_features_texts": [z1_text, z2_text, p1_text, p2_text],
            "logit_scale": self.base_model.logit_scale,
            "id_logits": [image_logits_1, text_logits_1, image_logits_2, text_logits_2],
            "mlm_scores": mlm_scores,
            "mae": [imgs, pred, mask]
        }



    