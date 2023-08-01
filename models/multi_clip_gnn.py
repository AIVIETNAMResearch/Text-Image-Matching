from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import  build_CLIP_from_openai_pretrained

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

class MultiViewCLIPGNN(nn.Module):
    def __init__(self, cfg):
        super(MultiViewCLIPGNN, self).__init__()
        self.base_model, _ = build_CLIP_from_openai_pretrained(cfg.MODEL.PRETRAINED, cfg.DATA.SIZE, cfg.MODEL.STRIDE)

        # SimSiam
        self.projector_image = projection_MLP(512, hidden_dim=512, out_dim=512)
        self.predictor_image = prediction_MLP(512, hidden_dim=512, out_dim=512)

        self.projector_text = projection_MLP(512, hidden_dim=512, out_dim=512)
        self.predictor_text = prediction_MLP(512, hidden_dim=512, out_dim=512)


    def encode_image(self, images):
        return self.base_model.encode_image(images)
    
    def encode_text(self, captions):
        return self.base_model.encode_text(captions)
    

    def forward(self, images1, images2, captions1, captions2):
        image_feats = self.encode_image(torch.cat([images1, images2], dim=0))
        text_feats = self.encode_text(torch.cat([captions1, captions2], dim=0))
        
        image_feats_1, image_feats_2 = torch.split(image_feats, images1.shape[0], dim=0)
        text_feats_1, text_feats_2 = torch.split(text_feats, captions1.shape[0], dim=0)
        
        # SimSiam
        z1_image = self.projector_image(image_feats_1)
        z2_image = self.projector_image(image_feats_2)
        p1_image = self.predictor_image(z1_image)
        p2_image = self.predictor_image(z2_image)

        z1_text = self.projector_text(text_feats_1)
        z2_text = self.projector_text(text_feats_2)
        p1_text = self.predictor_text(z1_text)
        p2_text = self.predictor_text(z2_text)

        return {
            "image_feats": [image_feats_1, image_feats_2],
            "text_feats": [text_feats_1, text_feats_2],
            "simsiam_features_images": [z1_image, z2_image, p1_image, p2_image],
            "simsiam_features_texts": [z1_text, z2_text, p1_text, p2_text],
            "logit_scale": self.base_model.logit_scale,
        }

