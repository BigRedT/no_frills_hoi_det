import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self,gamma=2,eps=1e-7,size_average=True):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average

    def forward(self,prob,labels):
        p_t = prob*labels + (1-prob)*(1-labels)
        loss = -((1.0-p_t)**self.gamma)*torch.log(p_t+self.eps)
        if self.size_average:
            loss = torch.mean(loss)
        return loss