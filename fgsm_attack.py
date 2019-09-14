import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def fgsm_attack_fun(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

#def fgsm_attack(model, image, target, device, epsilon=0.01, image_size=(128, 128)):

def fgsm_attack(model, image, target, device, epsilon=0.01, image_size=(128, 128)):
    steer = model(image)
    perturbed_image = image.clone()
    # steer = steer.type(torch.FloatTensor)
    # if (steer.item() > -0.1):
    #     target_steer = steer + target
    # else:
    #     target_steer = steer - target
    target_steer = steer - target
    target_steer = target_steer.to(device)
    image.requires_grad = True
    output = model(image)
    adv_output = output.clone()
    diff = 0
    # while abs(diff) < abs(target): 
    for i in range(5):
        loss = F.mse_loss(adv_output, target_steer)
        model.zero_grad()
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        perturbed_image = fgsm_attack_fun(perturbed_image, epsilon, image_grad)
        adv_output = model(perturbed_image)
        diff = abs(adv_output.detach().cpu().numpy() - output.detach().cpu().numpy())
    
    noise = torch.clamp(perturbed_image - image, 0, 1)

    return diff, perturbed_image, steer, adv_output, noise
if __name__ == "__main__":
    pass


