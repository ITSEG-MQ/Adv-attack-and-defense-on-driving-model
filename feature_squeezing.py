import matplotlib
# matplotlib.use('Agg')
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init
from data import UdacityDataset, Rescale, RandFlip, Preprocess, RandRotation, ToTensor, RandBrightness, RandRotateView, AdvDataset, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import math
import matplotlib.pyplot as plt
import csv
from os import path
from scipy.misc import imread, imresize, imsave
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
from fgsm_attack import fgsm_attack
from optimization_attack import optimized_attack
from experiment import fgsm_ex, opt_ex, opt_uni_ex, advGAN_ex, advGAN_uni_ex, generate_noise
from advGAN.models import Generator
import os
from advGAN_attack import advGAN_Attack
from scipy import ndimage

def reduce_bit(image, bit_size):
    image_int = np.rint(image * (math.pow(2, bit_size) - 1))
    image_float = image_int / (math.pow(2, bit_size) - 1)
    return image_float

def reduce_bit_pt(image, bit_size):
    image_int = torch.round(image * (math.pow(2, bit_size) - 1))
    image_float = image_int / (math.pow(2, bit_size) - 1)
    return image_float

def median_filter_np(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=2, mode='reflect')

def attack_detection(model_name, net, test_data_loader, attack, threshold=0.05):
    advGAN_generator = Generator(3,3, model_name).to(device)
    advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))         
    advGAN_generator.eval() 

    advGAN_uni_generator = Generator(3,3, model_name).to(device)
    advGAN_uni_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))    
    advGAN_uni_generator.eval()
    advGAN_uni_noise_seed = np.load(model_name + '_noise_seed.npy')


    opt_uni_noise = np.load(model_name + '_universal_attack_noise.npy')

    count_ori = 0
    count_adv = 0
    total = 0

    for _, example in enumerate(test_data_loader):
        # example = test_dataset[0]
        example_image = np.transpose(example['image'].squeeze(0).numpy(), (1, 2, 0))
        # example_image = example['image'].numpy()
        # squeeze_image = reduce_bit(example_image, 4)
        squeeze_image = median_filter_np(example_image, 2)
        example_image_tensor = torch.from_numpy(np.transpose(example_image, (-1, 0, 1))).unsqueeze(0)
        example_image_tensor = example_image_tensor.type(torch.FloatTensor)
        example_image_tensor = example_image_tensor.to(device)
        squeeze_image_tensor = torch.from_numpy(np.transpose(squeeze_image, (-1, 0, 1))).unsqueeze(0)
        squeeze_image_tensor = squeeze_image_tensor.type(torch.FloatTensor)
        squeeze_image_tensor = squeeze_image_tensor.to(device)
        if (abs(net(example_image_tensor) - net(squeeze_image_tensor)) > threshold):
            count_ori += 1

        # b = net(example_image_tensor)
        if attack == 'fgsm':
            _, perturbed_image, y_pred, y_adv, _ = fgsm_attack(net, example_image_tensor, target, device)
        elif attack == 'advGAN':
        # print(steer, adv_output)
            noise = advGAN_generator(example_image_tensor)
            perturbed_image = example_image_tensor + torch.clamp(noise, -0.3, 0.3)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            y_pred = net(example_image_tensor)
            y_adv = net(perturbed_image)
        elif attack == 'advGAN_uni':
            noise = advGAN_uni_generator(torch.from_numpy(advGAN_uni_noise_seed).type(torch.FloatTensor).to(device))
            perturbed_image = example_image_tensor + torch.clamp(noise, -0.3, 0.3)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            y_pred = net(example_image_tensor)
            y_adv = net(perturbed_image)
        elif attack == 'opt_uni':
            noise = torch.from_numpy(opt_uni_noise).type(torch.FloatTensor).to(device)
            perturbed_image = example_image_tensor + torch.clamp(noise, -0.3, 0.3)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            y_pred = net(example_image_tensor)
            y_adv = net(perturbed_image)
        elif attack == 'opt':
            perturbed_image, _, y_pred, y_adv = optimized_attack(net, target, example_image_tensor, device)
        # # print(net(perturbed_image))

        if abs(y_pred - y_adv) >= 0.3:
            total += 1
            perturbed_image_np = perturbed_image.squeeze().detach().cpu().numpy()
            perturbed_image_np = np.transpose(perturbed_image_np, (1, 2, 0))
            squeeze_perturbed_image = median_filter_np(perturbed_image_np, 2)
            squeeze_perturbed_image = torch.from_numpy(np.transpose(squeeze_perturbed_image, (-1, 0, 1))).unsqueeze(0)
            squeeze_perturbed_image = squeeze_perturbed_image.type(torch.FloatTensor)
            squeeze_perturbed_image = squeeze_perturbed_image.to(device)
            if (abs(net(perturbed_image) - net(squeeze_perturbed_image)) > threshold):
                count_adv += 1      
    print(attack, total, count_ori, count_adv)  

def plot_figure():
    fig = plt.figure(figsize=(12,4))
    df1 = pd.read_excel('feature_squeezing_epoch.xlsx', sheetname='Sheet1')
    df2 = pd.read_excel('feature_squeezing_nvidia.xlsx', sheetname='Sheet1')
    df3 = pd.read_excel('feature_squeezing_vgg16.xlsx', sheetname='Sheet1')
    ax1 = fig.add_subplot(1, 3, 1)
    df1.plot(ax=ax1, x='Threshold', y=["IT-FGSM", "Opt", "Opt_uni", "AdvGAN", "AdvGAN_uni", "Original(False)"],
     title='Detection rate on Epoch')
    plt.xticks([0.01, 0.05, 0.1, 0.15])
    plt.ylabel("Detection rate")
    ax2 = fig.add_subplot(1, 3, 2)
    df2.plot(ax=ax2, x='Threshold', y=["IT-FGSM", "Opt", "Opt_uni", "AdvGAN", "AdvGAN_uni", "Original(False)"],
     title='Detection rate on Nvidia')
    plt.xticks([0.01, 0.05, 0.1, 0.15])
    # plt.ylabel("Detection rate")
    ax3 = fig.add_subplot(1, 3, 3)
    df3.plot(ax=ax3, x='Threshold', y=["IT-FGSM", "Opt", "Opt_uni", "AdvGAN", "AdvGAN_uni", "Original(False)"],
     title='Detection rate on VGG16')
    plt.xticks([0.01, 0.05, 0.1, 0.15])
    # plt.ylabel("Detection rate")
    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend_.remove()
    # plt.ylabel("Detection rate")
    # ax1.legend(loc=2, bbox_to_anchor=(-1.0,1.0),borderaxespad = 0.)
    # plt.xticks([0.01, 0.05, 0.1, 0.15])
    # plt.ylabel("Detection rate")
    fig.tight_layout()
    # fig.show()
    plt.show()

def cal_detection_rate():
    for model_name in ['baseline', 'nvidia', 'vgg16']:
        # model_name = 'vgg16'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if  'baseline' in model_name:
            net = BaseCNN()
        elif  'nvidia' in model_name:
            net = Nvidia()
        elif 'vgg16' in model_name:
            net = Vgg16(False)

        net.load_state_dict(torch.load(model_name + '.pt')) 
        net.eval()
        net = net.to(device)
        dataset_path = '../udacity-data'
        root_dir = dataset_path
        test_composed = transforms.Compose([Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
        image_size = (128, 128)
        full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')
        full_indices = list(range(5614))
        test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
        train_indices = list(set(full_indices).difference(set(test_indices)))
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        test_data_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
        num_sample = len(full_dataset)
        target = 0.3
        # attack_detection(model_name, net, test_data_loader, attack='fgsm')
        # attack_detection(model_name, net, test_data_loader, attack='advGAN')
        # attack_detection(model_name, net, test_data_loader, attack='advGAN_uni')
        # attack_detection(model_name, net, test_data_loader, attack='opt_uni')
        # attack_detection(model_name, net, test_data_loader, attack='opt')
        print('threshold', 0.01)
        attack_detection(model_name, net, test_data_loader, attack='fgsm', threshold=0.01)
        attack_detection(model_name, net, test_data_loader, attack='advGAN', threshold=0.01)
        attack_detection(model_name, net, test_data_loader, attack='advGAN_uni', threshold=0.01)
        attack_detection(model_name, net, test_data_loader, attack='opt_uni', threshold=0.01)
        attack_detection(model_name, net, test_data_loader, attack='opt', threshold=0.01)

if __name__ == "__main__":
    plot_figure()

        # print()
        # print('threshold', 0.01)
        # attack_detection(model_name, net, test_data_loader, attack='fgsm', threshold=0.01)
        # attack_detection(model_name, net, test_data_loader, attack='advGAN', threshold=0.01)
        # attack_detection(model_name, net, test_data_loader, attack='advGAN_uni', threshold=0.01)
        # attack_detection(model_name, net, test_data_loader, attack='opt_uni', threshold=0.01)
        # attack_detection(model_name, net, test_data_loader, attack='opt', threshold=0.01)

            # print(net(squeeze_perturbed_image))
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(example_image)
            # plt.subplot(1, 2, 2)
            # plt.imshow(squeeze_image)
            # plt.show()
            # plt.savefig('111')