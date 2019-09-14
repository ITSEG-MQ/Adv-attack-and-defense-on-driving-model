import matplotlib
matplotlib.use('Agg')
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



def model_train(model_name, train=1, c=1, epochs=15):
    batch_size = 64
    lr = 0.0001     
    image_size=(128, 128)
    dataset_path = '../udacity-data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if  'baseline' in model_name:
        net = BaseCNN()
    elif  'nvidia' in model_name:
        net = Nvidia()
    elif 'vgg16' in model_name:
        net = Vgg16(False)
    
    net.apply(weight_init)
    net = net.to(device)
    # net.to(device)
    if train != 0:
        if train == 2:
            net.load_state_dict(torch.load(model_name + '.pt'))

        composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess('baseline'), ToTensor()])
        dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
        steps_per_epoch = int(len(dataset) / batch_size)

        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()

        optimizer = optim.Adam(net.parameters(), lr=lr)





        # x,y = train_generator.__next__()
        # print(x.shape)
        for epoch in range(epochs):
            total_loss = 0
            for step, sample_batched in enumerate(train_generator):
                if step <= steps_per_epoch:
                    batch_x = sample_batched['image']
                    # print(batch_x.numpy())
                    batch_y = sample_batched['steer']

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)


                    outputs = net(batch_x)
                    loss_x = criterion(outputs, batch_y)
                    loss_noise = 0
                    for i in range(25):
                        noise = torch.rand((batch_x.size(0), 3, 128, 128))
                        noise = noise.to(device)
                        noise -= 0.5
                        noise *= 0.1
                        noise_x = batch_x + noise
                        noise_y = net(noise_x)
                        loss_noise += criterion(noise_y, outputs)
                        # del noise
                        # del noise_x
                        # del noise_y
                    loss_noise /= 25
                    loss = loss_x + c * loss_noise
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss_x.item()
                    total_loss += running_loss
                else:
                    break
            print('Epoch %d  RMSE loss: %.4f' % (epoch,  total_loss / steps_per_epoch))
                
        torch.save(net.state_dict(), model_name + '.pt')
    else:
        net.load_state_dict(torch.load(model_name + '.pt'))



    net.eval()
    with torch.no_grad():
        yhat = []
        # test_y = []
        test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values
        # composed = transforms.Compose([Rescale(image_size),  Preprocess(model_name), ToTensor()])

        # dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
        # steps_per_epoch = int(len(dataset) / batch_size)

        # train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_composed = transforms.Compose([Rescale(image_size), Preprocess('baseline'), ToTensor()])
        test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for _,sample_batched in enumerate(test_generator):
            batch_x = sample_batched['image']
            # print(batch_x.size())
                # print(batch_x.size())
            batch_y = sample_batched['steer']
            # print(batch_y)

            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = net(batch_x)

            yhat.append(output.item())

        yhat = np.array(yhat)

        rmse = np.sqrt(np.mean((yhat-test_y)**2))
        print(rmse)
        plt.figure(figsize = (32, 8))
        plt.plot(test_y, 'r.-', label='target')
        plt.plot(yhat, 'b^-', label='predict')
        plt.legend(loc='best')
        plt.title("RMSE: %.2f" % rmse)
        # plt.show()
        model_fullname = "%s_%d.png" % (model_name, int(time.time()))
        plt.savefig(model_fullname)    

def attack_test(model_name, attack_train=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if  'baseline' in model_name:
        net = BaseCNN()
    elif  'nvidia' in model_name:
        net = Nvidia()
    elif 'vgg16' in model_name:
        net = Vgg16(False)

    net.load_state_dict(torch.load(model_name + '.pt')) 
    net = net.to(device)
    net.eval()
    dataset_path = '../udacity-data'
    root_dir = dataset_path
    test_composed = transforms.Compose([Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
    # test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    # test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # test_indices = list(np.random.choice(5614, int(0.1*5614), replace=False))
    # test_dataset = torch.utils.data.Subset(test_dataset, test_indices)


    image_size = (128, 128)
    full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')
    full_indices = list(range(5614))
    test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
    train_indices = list(set(full_indices).difference(set(test_indices)))
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    num_sample = len(test_dataset)
    target = 0.3

    if attack_train:
        attack_training(model_name, net, device, target, train_dataset)


    fgsm_ast, diff = fgsm_ex(test_data_loader, net, model_name, target, device, num_sample, image_size)
    print('fgsm', fgsm_ast)

    opt_ast, diff = opt_ex(test_dataset, net, model_name, target, device, num_sample, image_size)
    print('opt', opt_ast)

    optu_ast, diff = opt_uni_ex(test_data_loader, net, model_name, target, device, num_sample, image_size)
    print('optu', optu_ast)

    advGAN_ast, diff = advGAN_ex(test_data_loader, net, model_name, target, device, num_sample, image_size)        
    print('advGAN', advGAN_ast)

    advGAN_uni_ast, diff = advGAN_uni_ex(test_data_loader, net, model_name, target, device, num_sample, image_size)        
    print('advGAN_uni', advGAN_uni_ast)

def attack_training(model_name, net, device, target, train_dataset):
    print('Start universal attack training')
    perturbation = generate_noise(train_dataset, net, model_name, device, target)
    np.save(model_name + '_universal_attack_noise', perturbation)
    print('Finish universal attack training.')

    print('Start advGAN training')
    advGAN = advGAN_Attack(model_name, model_name + '.pt', target + 0.2, train_dataset)
    torch.save(advGAN.netG.state_dict(), './models/' + model_name +'_netG_epoch_60.pth')
    print('Finish advGAN training')
    advGAN_uni = advGAN_Attack(model_name, model_name + '.pt', target + 0.2, train_dataset, universal=True)
    advGAN_uni.save_noise_seed(model_name + '_noise_seed.npy')

if __name__ == "__main__":
    model_train('nvidia_regularization_0.1', train=1, epochs=15, c=0.1)
    attack_test('nvidia_regularization_0.1', attack_train=True)

    # model_train('baseline_regularization_0.5', train=1, c=0.5)
    # attack_test('baseline_regularization_0.5', attack_train=True)

    # model_train('baseline_regularization_0.1', train=1, c=0.1)
    # attack_test('baseline_regularization_0.1', attack_train=True)
    # model_train(train=2, epochs=5)