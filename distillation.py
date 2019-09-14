import matplotlib
matplotlib.use('Agg')
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init
from data import UdacityDataset, Rescale, RandFlip, Preprocess, RandRotation, ToTensor, RandBrightness, RandRotateView, AdvDataset, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import csv
from os import path
from scipy.misc import imread, imresize, imsave
import numpy as np 
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
from fgsm_attack import fgsm_attack
from optimization_attack import optimized_attack
import torch.nn as nn
from experiment import fgsm_ex, opt_ex, generate_noise
from adv_training import exp
import os
from advGAN_attack import advGAN_Attack 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

if __name__ == "__main__":
    models_name = ['nvidia', 'baseline', 'vgg16']
    image_size = (128, 128)
    batch_size = 32
    epochs = 15
    train = 1
    dataset_path = '../udacity-data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(model_name), ToTensor()])
    dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
    train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    steps_per_epoch = int(len(dataset) / batch_size)


    T = [5, 10, 15, 20, 25]
    #T = [25]
    for model_name in models_name:
        for t in T:
            distillation_net = Nvidia(type_='student')
            distillation_net = distillation_net.to(device)
            original_net = Nvidia(type_='student')
            original_net.load_state_dict(torch.load(model_name + '.pt'))
            original_net = original_net.to(device)
            original_net.eval()
            if train == 1:
                optimizer = optim.Adam(distillation_net.parameters(), lr=0.0001)
                alpha = 1
                for epoch in range(epochs):
                    total_loss = 0
                    for step, sample_batched in enumerate(train_generator):
                        if step <= steps_per_epoch:
                            batch_x = sample_batched['image']
                            batch_y = sample_batched['steer']
                            batch_x = batch_x.type(torch.FloatTensor)
                            batch_y = batch_y.type(torch.FloatTensor)
                            batch_x = batch_x.to(device)
                            batch_y = batch_y.to(device)
                            _, teacher_dist = original_net(batch_x)
                            output_steer, output_dist = distillation_net(batch_x)
                            loss_steer = nn.L1Loss()(output_steer, batch_y)
                            loss_dist = nn.MSELoss()(torch.exp(output_dist/t), torch.exp(teacher_dist/t))
                            loss = loss_steer + alpha * loss_dist
                            optimizer.zero_grad()

                            loss.backward()
                            optimizer.step()
                            running_loss = loss.item()
                            total_loss += running_loss
                        else:
                            break
                    #print('Epoch %d  RMSE loss: %.4f' % (epoch,  total_loss / steps_per_epoch))    
                torch.save(distillation_net.state_dict(), 'adv_training_models/' + model_name + '_distillation.pt')
                torch.save(distillation_net.state_dict(), 'adv_training_models/' + model_name + '_distillation_' + str(t) + '.pt')

            else:
                distillation_net.load_state_dict(torch.load('adv_training_models/' + model_name + '_distillation_' + str(t) + '.pt'))


            # original_net.type = 'teacher'
            original_net = None
            distillation_net.type = 'teacher'
            distillation_net.eval()
            advt_model = model_name + '_distillation'

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

                    output = distillation_net(batch_x)
                    # output = original_net(batch_x)


                    yhat.append(output.item())

                yhat = np.array(yhat)

                rmse = np.sqrt(np.mean((yhat-test_y)**2))
                print(rmse)
                # plt.savefig(model_fullname)
            
            full_indices = list(range(5614))
            test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
            train_indices = list(set(full_indices).difference(set(test_indices)))
            test_composed = transforms.Compose([Rescale((image_size[1],image_size[0])), Preprocess(), ToTensor()])

            full_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, type_='test')

            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            target = 0.3
            # generate opt uni noise on advt_model
            if not os.path.exists(advt_model + '_' + str(t) + '_universal_attack_noise.npy'):
                print('Start universal attack training')
                perturbation = generate_noise(train_dataset, distillation_net, advt_model, device, target)
                np.save( advt_model + '_' + str(t) + '_universal_attack_noise.npy', perturbation.detach().cpu().numpy())

            print('Finish universal attack training.')

            # advGAN training
            target_model_path = 'adv_training_models/' + advt_model  + '_' + str(t) + '.pt'
            # if not os.path.exists('./models/' + advt_model + '_netG_epoch_60.pth'):
            print('Start advGAN training')
            advGAN = advGAN_Attack(model_name, target_model_path, target + 0.2, train_dataset)
            torch.save(advGAN.netG.state_dict(), './models/' + advt_model + '_' + str(t) + '_netG_epoch_60.pth')

            print('Finish advGAN training')

            # advGAN_uni training
            # if not os.path.exists('./models/' + advt_model + '_universal_netG_epoch_60.pth'):
            print('Start advGAN_uni training')
            advGAN_uni = advGAN_Attack(advt_model,target_model_path, target + 0.2, train_dataset, universal=True)
            advGAN_uni.save_noise_seed(advt_model + '_' + str(t) + '_noise_seed.npy')
            torch.save(advGAN_uni.netG.state_dict(), './models/' + advt_model +  '_' + str(t) +'_universal_netG_epoch_60.pth')

            print('Finish advGAN_uni training')

            exp(distillation_net, model_name, 'distillation', test_dataset, device)