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


def exp(net, model_name, attack, test_dataset, device):
    original_net = None
    image_size = (128, 128)
    if model_name == 'baseline':
        original_net = BaseCNN()
    elif model_name == 'nvidia':
        original_net = Nvidia()
    elif model_name == 'vgg16':
        original_net = Vgg16()
    original_net.load_state_dict(torch.load(model_name + '.pt'))
    original_net = original_net.to(device)
    original_net.eval()

    # print(ast_ori, ast_dist)
    test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values
    test_composed = transforms.Compose([Rescale(image_size), Preprocess('baseline'), ToTensor()])
    test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    test_generator = DataLoader(test_dataset, batch_size=64, shuffle=False)
    target = 0.3
    ast_ori, _ = fgsm_ex(test_generator, original_net, model_name, target, device, len(test_dataset))
    ast_dist,_ = fgsm_ex(test_generator, net, model_name, target, device, len(test_dataset))
    print('fgsm:', ast_ori, ast_dist)

    advt_model = model_name + '_' + attack
    ast_ori, _ = advGAN_ex(test_generator, original_net, model_name, target, device, len(test_dataset))
    ast_dist,_ = advGAN_ex(test_generator, net, advt_model, target, device, len(test_dataset))
    print('advGAN:', ast_ori, ast_dist)

    advt_model = model_name + '_' + attack
    ast_ori, _ = advGAN_uni_ex(test_generator, original_net, model_name, target, device, len(test_dataset))
    ast_dist,_ = advGAN_uni_ex(test_generator, net, advt_model, target, device, len(test_dataset))
    print('advGAN_uni:', ast_ori, ast_dist)

    advt_model = model_name + '_' + attack
    ast_ori, _ = opt_uni_ex(test_generator, original_net, model_name, target, device, len(test_dataset))
    ast_dist,_ = opt_uni_ex(test_generator, net, advt_model, target, device, len(test_dataset))
    print('opt_uni:', ast_ori, ast_dist)

    ast_ori, _ = opt_ex(test_dataset, original_net, model_name, target, device, len(test_dataset))
    ast_dist,_ = opt_ex(test_dataset, net, model_name, target, device, len(test_dataset))
    print('opt:', ast_ori, ast_dist)

def test_on_gen(net, model_name, dataset_path, attack, device):
    original_net = None
    if model_name == 'baseline':
        original_net = BaseCNN()
    elif model_name == 'nvidia':
        original_net = Nvidia()
    elif model_name == 'vgg16':
        original_net = build_vgg16(False)
    original_net.load_state_dict(torch.load(model_name + '.pt'))
    original_net = original_net.to(device)
    original_net.eval()

    test_y = pd.read_csv('ch2_final_eval.csv')['steering_angle'].values
    test_composed = transforms.Compose([Rescale(image_size), Preprocess('baseline'), ToTensor()])
    test_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
    test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        # test on original dataset
        yhat = []
        y_original = []
        # test_y = []

        for _,sample_batched in enumerate(test_generator):
            batch_x = sample_batched['image']
            batch_y = sample_batched['steer']
            batch_x = batch_x.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = net(batch_x)
            output_ori = original_net(batch_x)
            # print(output.item(), batch_y.item())
            yhat.append(output.item())
            y_original.append(output_ori.item())
        yhat = np.array(yhat)
        y_original = np.array(y_original)
        rmse = np.sqrt(np.mean((yhat-test_y)**2))
        rmse_ori = np.sqrt(np.mean((y_original-test_y)**2))
        print('adv model on ori dataset:',rmse, 'ori model on ori dataset: ', rmse_ori)
        plt.figure(figsize = (32, 8))
        plt.plot(test_y, 'r.-', label='target')
        plt.plot(yhat, 'b^-', label='predict')
        plt.legend(loc='best')
        plt.title("RMSE: %.2f" % rmse)
        # plt.show()
        model_fullname = "%s_%d.png" % (model_name + '_' + attack, int(time.time()))
        plt.savefig(model_fullname)
    
    test_generator = DataLoader(test_dataset, batch_size=64, shuffle=False)    
    target = 0.3

    # test adv_training model on adv images generated based on itself
    # ast_ori, _ = fgsm_ex(test_generator, original_net, 'baseline', target, device, len(test_dataset))
    # ast_dist,_ = fgsm_ex(test_generator, net, 'baseline', target, device, len(test_dataset))
    # print(ast_ori, ast_dist)
    success = 0
    success_ = 0
    # test  adv_training model on adv images generated based on original model
    for _,sample_batched in enumerate(test_generator):
        batch_x = sample_batched['image']
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        y_pred = net(batch_x)
        y_pred_ori = original_net(batch_x)
        # fgsm_attack
        adv_x = fgsm_attack_(original_net, batch_x, target, device)
        y_fgsm = net(adv_x)
        y_ori_fgsm = original_net(adv_x)
        diff = abs(y_fgsm - y_pred)
        success += len(diff[diff >= abs(target)])

        diff = abs(y_ori_fgsm - y_pred_ori)
        success_ += len(diff[diff >= abs(target)])
    print('fgsm', success / len(test_dataset), success_ / len(test_dataset))
    
    # opt attack
    # success = 0
    # success_ = 0
    # for _,sample_batched in enumerate(test_dataset):
    #     batch_x = sample_batched['image']
    #     batch_x = batch_x.type(torch.FloatTensor)
    #     batch_x = batch_x.unsqueeze(0)
    #     batch_x = batch_x.to(device)
    #     y_pred = net(batch_x)
    #     y_pred_ori = original_net(batch_x)
    #     # fgsm_attack
    #     # adv_x = fgsm_attack_(original_net, batch_x, target, device)
    #     adv_x,_,_,_ = optimized_attack(original_net, target, batch_x)
    #     y_fgsm = net(adv_x)
    #     y_ori_fgsm = original_net(adv_x)
    #     diff = abs(y_fgsm - y_pred)
    #     success += len(diff[diff >= abs(target)])

    #     diff = abs(y_ori_fgsm - y_pred_ori)
    #     success_ += len(diff[diff >= abs(target)])
    # print('opt', success / len(test_dataset), success_ / len(test_dataset))
    
    # opt universal attack
    noise_u = np.load(model_name + '_universal_attack_noise.npy')
    noise_u = torch.from_numpy(noise_u).type(torch.FloatTensor).to(device)
    success = 0
    success_ = 0    
    for _,sample_batched in enumerate(test_generator):
        batch_x = sample_batched['image']
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        y_pred = net(batch_x)
        y_pred_ori = original_net(batch_x)

        # adv_x = fgsm_attack_(original_net, batch_x, target, device)
        # noise = advGAN_generator(batch_x)
        perturbed_image = batch_x + noise_u
        adv_x = torch.clamp(perturbed_image, 0, 1)
        y_fgsm = net(adv_x)
        y_ori_fgsm = original_net(adv_x)
        diff = abs(y_fgsm - y_pred)
        success += len(diff[diff >= abs(target)])

        diff = abs(y_ori_fgsm - y_pred_ori)
        success_ += len(diff[diff >= abs(target)])
    print('opt uni', success / len(test_dataset), success_ / len(test_dataset))

    # test for advGAN attack
    success = 0
    success_ = 0
    advGAN_generator = Generator(3,3, model_name).to(device)
    advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))    

    for _,sample_batched in enumerate(test_generator):
        batch_x = sample_batched['image']
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        y_pred = net(batch_x)
        y_pred_ori = original_net(batch_x)

        # adv_x = fgsm_attack_(original_net, batch_x, target, device)
        noise = advGAN_generator(batch_x)
        perturbed_image = batch_x + torch.clamp(noise, -0.3, 0.3)
        adv_x = torch.clamp(perturbed_image, 0, 1)
        y_fgsm = net(adv_x)
        y_ori_fgsm = original_net(adv_x)
        diff = abs(y_fgsm - y_pred)
        success += len(diff[diff >= abs(target)])

        diff = abs(y_ori_fgsm - y_pred_ori)
        success_ += len(diff[diff >= abs(target)])
    print('advGAN', success / len(test_dataset), success_ / len(test_dataset))

    # test for advGAN uni attack

    advGAN_uni_generator = Generator(3,3, model_name).to(device)
    advGAN_uni_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))    
    noise_seed = np.load(model_name + '_noise_seed.npy')
    noise_a = advGAN_uni_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
    success = 0
    success_ = 0    
    for _,sample_batched in enumerate(test_generator):
        batch_x = sample_batched['image']
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        y_pred = net(batch_x)
        y_pred_ori = original_net(batch_x)

        # adv_x = fgsm_attack_(original_net, batch_x, target, device)
        # noise = advGAN_generator(batch_x)
        perturbed_image = batch_x + torch.clamp(noise_a, -0.3, 0.3)
        adv_x = torch.clamp(perturbed_image, 0, 1)
        y_fgsm = net(adv_x)
        y_ori_fgsm = original_net(adv_x)
        diff = abs(y_fgsm - y_pred)
        success += len(diff[diff >= abs(target)])

        diff = abs(y_ori_fgsm - y_pred_ori)
        success_ += len(diff[diff >= abs(target)])
    print('advGAN uni', success / len(test_dataset), success_ / len(test_dataset))
    # print(success / len(test_dataset), success_ / len(test_dataset))
    # y_adv_hat = np.array(y_adv_hat)
    # print(y_adv_hat)
    # rmse_fgsm = np.sqrt(np.mean((np.array(y_adv['fgsm_attack']-test_y)**2)))
    # plt.figure(figsize = (32, 8))
    # plt.plot(test_y, 'r.-', label='target')
    # plt.plot(np.array(y_adv['fgsm_attack']), 'b^-', label='predict_adv_fgsm')
    # plt.legend(loc='best')
    # plt.title("RMSE: %.4f" % rmse_fgsm)
    # # plt.show()
    # model_fullname = "%s_%d.png" % (model_name + '_adv_fgsm', int(time.time()))
    # plt.savefig(model_fullname)    
    # rmse_opt = np.sqrt(np.mean((np.array(y_adv['opt_attack']-test_y)**2)))
    # plt.figure(figsize = (32, 8))
    # plt.plot(test_y, 'r.-', label='target')
    # plt.plot(np.array(y_adv['opt_attack']), 'b^-', label='predict_adv_opt')
    # plt.legend(loc='best')
    # plt.title("RMSE: %.4f" % rmse_opt)
    # # plt.show()
    # model_fullname = "%s_%d.png" % (model_name + '_adv_opt', int(time.time()))
    # plt.savefig(model_fullname)

    # rmse_ = np.sqrt(np.mean((yhat-test_y)**2))
    # rmse_opt_adv =  np.sqrt(np.mean((y_opt_hat-test_y)**2))
    # print('adv model on fgsm adv dataset: ', rmse_fgsm, 'adv model on opt adv dataset: ', rmse_opt)
    # print('ori model on fgsm adv dataset: ', rmse_, 'ori model on opt adv dataset: ', rmse_opt_adv)
    # plt.figure(figsize = (32, 8))
    # plt.plot(test_y, 'r.-', label='target')
    # plt.plot(yhat, 'b^-', label='predict')
    # plt.legend(loc='best')
    # plt.title("RMSE: %.2f" % rmse_)
    # # plt.show()
    # model_fullname = "%s_%d.png" % (model_name, int(time.time()))
    # plt.savefig(model_fullname)        
def fgsm_attack_(model, image, target, device, epsilon=0.01, image_size=(128, 128)):
    # image = image.unsqueeze(0)
    image =image.type(torch.FloatTensor)
    image = image.to(device)
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
        perturbed_image = fgsm_attack(perturbed_image, epsilon, image_grad)
        adv_output = model(perturbed_image)
        diff = abs(adv_output.detach().cpu().numpy() - output.detach().cpu().numpy())
    # diff, perturbed_image, steer, adv_output = fgsm_attack(model, image, target, device, epsilon=epsilon, image_size=image_size)
    # noise = torch.clamp(perturbed_image - image, 0, 1)
    return perturbed_image

if __name__ == "__main__":
    batch_size = 32
    lr = 0.0001
    epochs = 15
    # train all models 
    train = 1
    test = 0
    resized_image_height = 128
    resized_image_width = 128
    image_size=(resized_image_width, resized_image_height)
    # dataset_path = args.root_dir
    # adv_dataset_path = args.adv_root_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # models_name = ['baseline', 'nvidia', 'vgg16']
    models_name = ['baseline','nvidia','vgg16']

    adv_datasets = '../udacity-data/adv_data'
    #attacks = [ 'universal_attack',  'advGAN_universal_attack']
    #attacks = ['opt_attack', 'fgsm_attack','advGAN_attack','universal_attack',  'advGAN_universal_attack']
    attacks = ['fgsm_attack']
    dataset_path = '../udacity-data/'
    if test:
        full_indices = list(range(5614))
        test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
        train_indices = list(set(full_indices).difference(set(test_indices)))
        test_composed = transforms.Compose([Rescale((image_size[1],image_size[0])), Preprocess(), ToTensor()])

        full_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, type_='test')

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        net = Vgg16()
        # net.load_state_dict(torch.load('adv_training_models/' + 'nvidia' + '_' + 'fgsm_attack' +  '.pt'))
        net.load_state_dict(torch.load('adv_training_models/vgg16_fgsm_attack.pt'))

        net = net.to(device)
        net.eval()
        #test_on_gen(net, 'baseline', dataset_path, 'fgsm', device)
        exp(net, 'vgg16', 'fgsm_attack', test_dataset, device)
    if train:
        full_indices = list(range(5614))
        test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
        train_indices = list(set(full_indices).difference(set(test_indices)))
        test_composed = transforms.Compose([Rescale((image_size[1],image_size[0])), Preprocess(), ToTensor()])

        full_dataset = UdacityDataset(dataset_path, ['testing'], test_composed, type_='test')

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        for model_name in models_name:
            if model_name == 'baseline':
                net = BaseCNN()
            elif model_name == 'nvidia':
                net = Nvidia()
            elif model_name == 'vgg16':
                net = Vgg16()

            net = net.to(device)
            for attack in attacks:
                print(model_name, attack)
                adv_dataset_path = adv_datasets + '/' + model_name + '/' + attack
            
                if not os.path.exists(adv_dataset_path):
                    os.mkdir(adv_dataset_path)
                advt_model = model_name + '_' + attack
                if train != 0:
                    if train == 2:
                        net.load_state_dict(torch.load('adv_training_models/' + advt_model +  '.pt'))
                        #net.load_state_dict(torch.load(model_name + '.pt'))
                    
                    else:
                        composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(model_name), ToTensor()])
                        dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)

                        # adv_composed = transforms.Compose([RandFlip(), RandRotation(),  Preprocess(model_name), ToTensor()])
                        adv_dataset = AdvDataset(adv_dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'])
                        concat_dataset = ConcatDataset(dataset, adv_dataset)
                        steps_per_epoch = int(len(concat_dataset) / batch_size)

                        train_generator = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

                        criterion = nn.L1Loss()

                        optimizer = optim.Adam(net.parameters(), lr=lr)

                        for epoch in range(epochs):
                            total_loss = 0
                            for step, sample_batched in enumerate(train_generator):

                                if step <= steps_per_epoch:
                                    batch1_x = sample_batched[0]['image']
                                    batch1_y = sample_batched[0]['steer']

                                    batch1_x = batch1_x.type(torch.FloatTensor)
                                    batch1_y = batch1_y.type(torch.FloatTensor)
                                    batch1_x = batch1_x.to(device)
                                    batch1_y = batch1_y.to(device)
                                    batch2_x = sample_batched[1]['image']

                                    batch2_y = sample_batched[1]['steer']

                                    batch2_x = batch2_x.type(torch.FloatTensor)
                                    batch2_y = batch2_y.type(torch.FloatTensor)
                                    batch2_x = batch2_x.to(device)
                                    batch2_y = batch2_y.to(device)

                                    batch_x = torch.cat((batch1_x, batch2_x), 0)
                                    batch_y = torch.cat((batch1_y, batch2_y), 0)
                                    batch_x = batch_x.to(device)
                                    batch_y = batch_y.to(device)

                                    outputs = net(batch_x)
                                    loss = criterion(outputs, batch_y)
                                    optimizer.zero_grad()

                                    loss.backward()
                                    optimizer.step()
                                    running_loss = loss.item()
                                    total_loss += running_loss
                                else:
                                    break
                            print('Epoch %d  RMSE loss: %.4f' % (epoch,  total_loss / steps_per_epoch))
                                
                        torch.save(net.state_dict(), 'adv_training_models/' + advt_model + '.pt')

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
                    test_dataset_ = UdacityDataset(dataset_path, ['testing'], test_composed, 'test')
                    test_generator = DataLoader(test_dataset_, batch_size=1, shuffle=False)
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
                    model_fullname = "%s_%s_%d.png" % (model_name, attack, int(time.time()))
                    # plt.savefig(model_fullname)
                
                target = 0.3
                # generate opt uni noise on advt_model
                #if not os.path.exists(advt_model + '_universal_attack_noise.npy'):
                print('Start universal attack training')
                perturbation = generate_noise(train_dataset, net, advt_model, device, target)
                np.save(advt_model + '_universal_attack_noise', perturbation.detach().cpu().numpy())
                print('Finish universal attack training.')

                # advGAN training
                target_model_path = 'adv_training_models/' + advt_model + '.pt'
                #if not os.path.exists('./models/' + advt_model + '_netG_epoch_60.pth'):
                print('Start advGAN training')
                advGAN = advGAN_Attack(model_name, target_model_path, target + 0.2, train_dataset)
                torch.save(advGAN.netG.state_dict(), './models/' + advt_model + '_netG_epoch_60.pth')

                print('Finish advGAN training')

                # advGAN_uni training
                #if not os.path.exists('./models/' + advt_model + '_universal_netG_epoch_60.pth'):
                print('Start advGAN_uni training')
                advGAN_uni = advGAN_Attack(advt_model,target_model_path, target + 0.2, train_dataset, universal=True)
                torch.save(advGAN_uni.netG.state_dict(), './models/' + advt_model + '_universal_netG_epoch_60.pth')
                print('Finish advGAN_uni training')

                exp(net, model_name, attack, test_dataset, device)