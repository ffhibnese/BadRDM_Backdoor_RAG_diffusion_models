import torch
from torch.nn.functional import cosine_similarity
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import numpy as np

class SimilarityLoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)

        loss = -1 * cosine_similarity(input, target, dim=1)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
class MSELoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.loss_fkt = torch.nn.MSELoss(reduction=reduction)
        self.flatten = flatten

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)
        loss = self.loss_fkt(input, target)
        return loss
    
class ContrastLoss(torch.nn.Module):
    
    def __init__(self, t=1.0):
        super().__init__()
        self.t = t
        
    def forward(self, txt_embed: torch.Tensor, posioned_img: torch.Tensor, clean_img: torch.Tensor):
        criterion = CrossEntropyLoss()
        B, C = txt_embed.shape
        assert txt_embed.shape == posioned_img.shape

        posioned_img = posioned_img / posioned_img.norm(dim=1, keepdim=True)
        clean_img = clean_img / clean_img.norm(dim=1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(dim=1, keepdim=True)
        if posioned_img.dtype == torch.float16:
            posioned_img = posioned_img.float()
        
        pos_logits = torch.bmm(txt_embed.view(B, 1, C), posioned_img.view(B, C, 1)).squeeze(1)
        neg_logits = torch.mm(txt_embed, clean_img.T)
        
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits /= self.t
        
        gt = torch.zeros(B, device=txt_embed.device)
        loss = criterion(logits, gt.long())
        return loss
    
class ClipLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, txt_embed: torch.Tensor, img_embed: torch.Tensor):
        B = img_embed.shape[0]
        loss_img = CrossEntropyLoss()
        loss_txt = CrossEntropyLoss()
        
        assert txt_embed.shape == img_embed.shape
        
        img_embed = img_embed / img_embed.norm(dim=1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_embed @ txt_embed.t()
        logits_per_text = logits_per_image.t()
        
        gt = torch.arange(B, device=img_embed.device).long()
        loss = (loss_img(logits_per_image, gt) + loss_txt(logits_per_text, gt)) / 2
        return loss
