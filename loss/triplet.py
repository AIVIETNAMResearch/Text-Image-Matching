import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_neg_mask(labels):

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


# Triplet loss + DistanceWeight Miner
# Sampling Matters in Deep Embedding Learning, ICCV, 2017
# more information refer to https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer
class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.2, ):
        super().__init__()

        self.opt = opt
        self.margin = margin
        
        self.cut_off = 0.5
        self.d = 512
        self.nonzero_loss_cutoff = 1.7
        
    def forward(self, im, s, img_ids):

        sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
            #print(pos_mask.sum())
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())
        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
     
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)   
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  
        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum() 

        return loss