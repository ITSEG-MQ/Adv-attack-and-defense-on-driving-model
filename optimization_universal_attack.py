import os
from optimization_attack import optimized_attack
import numpy as np 
np.random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data import UdacityDataset, Rescale, Preprocess, ToTensor
from model import BaseCNN
from viewer import draw
from scipy.misc import imresize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    v_ = v.detach().cpu().numpy()
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v_ = v_ * min(1, xi/np.linalg.norm(v_.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v_ = np.sign(v_) * np.minimum(abs(v_), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return torch.from_numpy(v_)

def universal_attack(dataset, model, device, target, delta=0.3, max_iters = np.inf, xi=10, p=np.inf, max_iter_lbfgs=30):
    v = 0
    fooling_rate = 0.0
    num_images = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,1,True)    

    itr = 0
    while fooling_rate < 1-delta and itr < max_iters:
        # np.random.shuffle(dataset)
        print('Starting pass number: ', itr)

        for k, data in enumerate(dataloader):
            cur_img = data['image']
            cur_img = cur_img.type(torch.FloatTensor)
            cur_img = cur_img.to(device)
            perturbed_image = cur_img + v
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = perturbed_image.to(device)
            temp = is_adversary(model, cur_img, perturbed_image, target)
            if not temp[0]:
                # print('>> k = ', k, ', pass #', itr)
                _, d_noise, _, _ = optimized_attack(model, temp[1], perturbed_image, device)
                v = v + d_noise
                v = proj_lp(v, xi, p)
                v = v.to(device)

        
        itr += 1

        count = 0
        for _, data in enumerate(dataloader):
            cur_img = data['image']
            cur_img = cur_img.type(torch.FloatTensor)
            cur_img = cur_img.to(device)
            perturbed_image = cur_img + v
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            perturbed_image = perturbed_image.to(device)            
            if (is_adversary(model, cur_img, perturbed_image, target)[0]):
                count += 1
        fooling_rate = count / num_images

        print('Fooling rate: ', fooling_rate)
    # demension of v : (1, 3, image_size)    
    return v

def is_adversary(model, x, x_adv, target):
    # print(target)
    y_pred = model(x).item()
    y_adv = model(x_adv).item()
    # print(y_pred, y_adv)
    if (abs(y_adv - y_pred) >= abs(target)):
        return [True]
    else:
        return [False, target - (y_adv - y_pred)]

def is_adversary_(model, x, x_adv, target):
    y_pred = model.predict(x)
    y_adv = model.predict(x_adv)
    if (abs(y_adv - y_pred) >= abs(target)):
        return [True]
    else:
        return [False, target - (y_adv - y_pred)]

def generate_noise(dataset, model, model_name, device, target):
    perturbation = universal_attack(dataset, model, device, target)
    perturbation = perturbation.detach().cpu().numpy()
    return perturbation

if __name__ =="__main__":
    target_model = 'baseline.pt'
    target = 0.3
    root_dir = '../udacity-data'
    test_composed = transforms.Compose([Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
    full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')
    train_size = int(0.8*len(full_dataset))
    test_size =len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    dataloader = torch.utils.data.DataLoader(train_dataset,1,True)    

    model = BaseCNN()
    model.to(device)
    model.load_state_dict(torch.load(target_model))
    model.eval()

    perturbation = universal_attack(dataloader, model, device, target, 0.3)
    np.save('universal_attack_noise.npy', perturbation.detach().cpu().numpy())
    # i = 0
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,1,True)
    # for _, data in enumerate(test_dataloader):
    #     image = data['image']
    #     perturbed_image = image + perturbation
    #     image =image.type(torch.FloatTensor)
    #     perturbed_image = perturbed_image.type(torch.FloatTensor)
    #     image = image.to(device)
    #     perturbed_image = perturbed_image.to(device)
    #     y_pred = model(image)
    #     y_adv = model(perturbed_image)

    # draw_on_image(model, x, y, image_name, peturbation)
    # peturbation = peturbation.reshape(image_size)
    # imsave('universal_peturbation.jpg', np.clip(peturbation, 0, 1))
