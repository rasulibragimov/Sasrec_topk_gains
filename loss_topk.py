import torch
import torch.nn.functional as F
import torch.nn as nn

# Documentation to be added

def top1_loss(pos, neg):
    sigm = torch.nn.Sigmoid()
    return (sigm(neg - pos) + sigm(neg * neg)).sum()/len(neg)

def bpr_max_loss(pos, neg, soft_neg):
    sigm = torch.nn.Sigmoid()
    return (-torch.log(soft_neg * sigm(pos - neg))).sum()/len(neg)

def top1_max_loss(pos, neg, soft_neg):
    sigm = torch.nn.Sigmoid()
    return (soft_neg*(sigm(neg - pos) + sigm(neg * neg))).sum()/len(neg)