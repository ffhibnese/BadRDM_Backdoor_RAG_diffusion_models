import numpy as np
import torch
import torch.nn as nn
import copy
from torchvision import transforms

class ImageAttacker():
    def __init__(self, normalization, eps=2/255, steps=150, step_size=0.5/255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        
    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)    
    
    # Pull image embeddings toward the target text embedding.
    def loss_func(self, adv_imgs_embeds, txts_embeds, norm=True):  
        if norm:
            # normalize adv image embeddings
            adv_imgs_embeds = adv_imgs_embeds / torch.norm(adv_imgs_embeds.detach(), dim=1, keepdim=True)

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        
        loss_IaTcpos = it_sim_matrix.sum(-1).mean() # changed
        loss = loss_IaTcpos
        # print(f'sim: {loss}')
        
        return loss
    
    # Optimize images toward the text embedding.
    def attack(self, model, imgs, device, scales=None, txt_embeds=None):
        if scales is not None:
            scales = [float(scale) for scale in scales.split(',')]
            
        model.eval()
        b = imgs.shape[0]
        scales_num = 1 if scales is None else len(scales) + 1
        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        for i in range(self.steps):
            print(f'the {i+1} time')
            adv_imgs.requires_grad_()
            scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device)
            
            if self.normalization is not None:
                adv_imgs_embeds = model.encode_image(self.normalization(scaled_imgs))
            else:
                adv_imgs_embeds = model.encode_image(scaled_imgs)
            
            model.zero_grad()
            with torch.enable_grad():
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    loss_item = self.loss_func(adv_imgs_embeds[i*b:i*b+b], txt_embeds.detach())
                    loss += loss_item
            loss.backward()
            
            grad = adv_imgs.grad
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            
            # PGD
            perturbation = self.step_size * grad.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs-self.eps), imgs+self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            # sim = self.loss_func(adv_imgs_embeds=model.encode_image(adv_imgs.detach()), txts_embeds=txt_embeds.detach())
            # print(f'sim: {sim}')
            
        return adv_imgs
