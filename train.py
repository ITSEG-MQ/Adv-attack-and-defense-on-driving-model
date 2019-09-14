import matplotlib
matplotlib.use('Agg')
from model import BaseCNN, Nvidia, Vgg16, build_vgg16, weight_init
from data import UdacityDataset, Rescale, RandFlip, Preprocess, RandRotation, ToTensor, RandBrightness, RandRotateView
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
from adv_training import test_on_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training.')
    parser.add_argument("--model_name", type=str, default="baseline")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    model_name = args.model_name
    camera = 'center'
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    resized_image_height = 0
    resized_image_width = 0
    train = args.train
    resized_image_height = 128
    resized_image_width = 128
    # if model_name == "baseline":
    #     resized_image_height = 128
    #     resized_image_width = 128
    # elif model_name == "vgg16":
    #     resized_image_height = 224
    #     resized_image_width = 224
    # elif model_name == "nvidia":
    #     resized_image_height = 66
    #     resized_image_width = 200        
    image_size=(resized_image_width, resized_image_height)
    dataset_path = args.root_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if model_name == 'baseline':
        net = BaseCNN()
    elif model_name == 'nvidia':
        net = Nvidia()
    elif model_name == 'vgg16':
        net = Vgg16()
    
    net.apply(weight_init)
    net = net.to(device)
    # net.to(device)
    if train != 0:
        if train == 2:
            net.load_state_dict(torch.load(model_name + '.pt'))

        composed = transforms.Compose([Rescale(image_size), RandFlip(), RandRotation(),  Preprocess(model_name), ToTensor()])
        dataset = UdacityDataset(dataset_path, ['HMB1', 'HMB2', 'HMB4', 'HMB5','HMB6'], composed)
        steps_per_epoch = int(len(dataset) / batch_size)

        train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        if model_name == 'vgg16':
            optimizer = optim.Adam(net.parameters(), lr=lr)
        else:
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
                    
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()
                    total_loss += running_loss
                else:
                    break
            print('Epoch %d  RMSE loss: %.4f' % (epoch,  total_loss / steps_per_epoch))
                
        torch.save(net.state_dict(), model_name + '_.pt')
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
        # plt.savefig(model_fullname)
    
    #test_on_file(net, model_name, dataset_path, device)