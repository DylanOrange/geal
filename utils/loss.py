
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def kl_div(p_out, q_out, get_softmax=True):
    KLD = nn.KLDivLoss(reduction='batchmean')
    B = p_out.size(0)

    if get_softmax:
        p_out = F.softmax(p_out.view(B,-1),dim=-1)
        q_out = F.log_softmax(q_out.view(B,-1),dim=-1)

    kl_loss = KLD(q_out, p_out)

    return kl_loss

class HM_Loss(nn.Module):
    def __init__(self):
        super(HM_Loss, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, pred, target):
        #[B, N, 18]
        pred = pred.unsqueeze(-1)
        target = target.unsqueeze(-1)
        temp1 = -(1-self.alpha)*torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -self.alpha*torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)
        return CELoss+3.0*DICELoss

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, pred, target):
        # Compute cosine similarity along the last dimension (N)
        cosine_sim = F.cosine_similarity(pred, target, dim=1)  # Shape: (B,)

        # Convert cosine similarity to cosine similarity loss
        loss = 1 - cosine_sim  # Shape: (B,)
        # Take the mean of the loss across the batch to get a scalar loss
        return loss.mean()
    

class SIM_Loss(nn.Module):
    def __init__(self):
        super(SIM_Loss, self).__init__()

        self.eps = 1e-12
        self.criterion_bce = nn.BCELoss()

    def forward(self, pred, target):
        
        normal_pred, normal_target = pred/(pred.sum()+self.eps), target/(target.sum() + self.eps)

        intersection = torch.minimum(normal_pred, normal_target).sum()

        return 1-intersection
        
class CrossModalCenterLoss(nn.Module):
    """Center loss.    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim=512, local_rank=None):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.local_rank = local_rank

        if self.local_rank != None:
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cuda')
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        temp = torch.mm(x, self.centers.t())
        distmat = distmat - 2*temp

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def l1_loss(network_output, gt, mask):
    masked_loss = torch.abs(network_output - gt) * mask
    loss = masked_loss.sum() / (mask.sum() + 1e-8)

    # loss = torch.abs(network_output - gt).mean()
    return loss

def cal_kl_loss(fmap1, fmap2):

    log_prob1 = F.log_softmax(fmap1, dim=1)  # Log probabilities for fmap1
    prob2 = F.softmax(fmap2, dim=1)          # Probabilities for fmap2

    # Compute the KL divergence
    kl_loss = F.kl_div(log_prob1, prob2, reduction='batchmean')

    return kl_loss
