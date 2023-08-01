from .triplet import TripletLoss
from .crossentropy import cross_entropy
import torch.nn.functional as F
import torch
from .sdm import compute_sdm
from .simsiam import SimsiamLoss
from .focal import FocalLoss
import torch.nn as nn

def compute_clip(text_features, image_features, pid, logit_scale, batch_size, loss_margin=0.):
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    sim_i_2_t = torch.matmul(image_features, torch.t(text_features))
    acc_sim = sim_i_2_t.clone().detach()
    sim_i_2_t = sim_i_2_t - (torch.eye(batch_size).cuda() * loss_margin)
    sim_i_2_t = torch.mul(logit_scale, sim_i_2_t)
    sim_t_2_i = sim_i_2_t.t()
    loss_t_2_i = F.cross_entropy(sim_t_2_i, labels.cuda())
    loss_i_2_t = F.cross_entropy(sim_i_2_t, labels.cuda())
    clip_loss = (loss_t_2_i+loss_i_2_t)/2

    return clip_loss

def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_triplet(image_features, text_features, pid, margin=0.2):
    criterion = TripletLoss(margin=margin)

    loss = criterion(image_features, text_features, pid.cuda()) 
    return loss 

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)