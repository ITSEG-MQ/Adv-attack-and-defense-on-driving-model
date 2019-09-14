import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

def optimized_attack(target_model, target, x, device):
    y_pred = target_model(x)
    y_adv = y_pred
    # if y_pred > -0.1:
    #     y_target = y_pred - target
    # else:
    #     y_target = y_pred + target
    y_target = y_pred + target

    perturb = torch.zeros_like(x)
    perturb.requires_grad = True
    perturb = perturb.to(device)
    optimizer = optim.Adam(params=[perturb], lr=0.005)
    diff = 0

    # while abs(diff) < abs(target):
    for i in range(100):
        perturbed_image = x + perturb
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        y_adv = target_model(perturbed_image)
        optimizer.zero_grad()
        loss_y = F.mse_loss(y_adv, y_target)
        loss_n = torch.mean(torch.pow(perturb, 2))
        loss_adv = loss_y + loss_n
        loss_adv.backward(retain_graph=True)
        optimizer.step()
        diff = y_adv.item() - y_pred.item()
        if abs(diff) >= abs(target):
            break
        # print(diff, target)


    
    return perturbed_image, perturb, y_pred, y_adv