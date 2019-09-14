from functools import wraps
import time
import cv2
from scipy.misc import imsave, imread
from torch.utils.data import DataLoader
import os
from advGAN_attack import advGAN_Attack
from optimization_universal_attack import generate_noise
import attack_test
from optimization_attack import optimized_attack
from advGAN.models import Generator
from fgsm_attack import fgsm_attack
from torchvision import datasets, transforms
from scipy.misc import imresize
from viewer import draw
from model import BaseCNN, Nvidia, build_vgg16, Vgg16
from data import UdacityDataset, Rescale, Preprocess, ToTensor
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.func_name, str(t1-t0))
              )
        return result
    return function_timer


def fgsm_test(model, image, target, device):
    _, perturbed_image_fgsm, _, adv_output_fgsm, noise_fgsm = fgsm_attack(
        model, image, target, device)


def opt_test(model, image, target, device):
    perturbed_image_opt, noise_opt, _, adv_output_opt = optimized_attack(
        model, target, image, device)
    return
# def advGAN_test(model, image, generator, device):


def optu_test(model, image, noise, device):
    perturbed_image_optu = image + noise
    perturbed_image_optu = torch.clamp(perturbed_image_optu, 0, 1)
    adv_output_optu = model(perturbed_image_optu)
    perturbed_image_optu = perturbed_image_optu.squeeze(
        0).detach().cpu().numpy().transpose(1, 2, 0)


if __name__ == "__main__":
    model = BaseCNN()
    model_name = 'baseline'
    model.load_state_dict(torch.load('baseline.pt'))
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    model.eval()
    target = 0.3
    # image = imread(
    #     '/media/dylan/Program/cg23-dataset/testing/center/1479425441182877835.jpg')[200:, :]
    # image = imresize(image, (128, 128))
    # image = image / 255.
    # image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
    # image =image.type(torch.FloatTensor)
    # image = image.to(device)
    # output = model(image)



    # fgsm

    # opt

    # optu

    # t0 = time.time()
    # # optu_test(model, image, noise_optu, device)
    # # fgsm_test(model, image, target, device)
    # opt_test(model, image, target, device)
    # t1 = time.time()
    # print ("Total time running: %s seconds" %
    #     (str(t1-t0)))
    # advGAN
    # advGAN_generator.load_state_dict(torch.load(
    #     './models/' + model_name + '_netG_epoch_60.pth'))
    # noise_advGAN = advGAN_generator(image)
    # perturbed_image_advGAN = image + torch.clamp(noise_advGAN, -0.3, 0.3)
    # perturbed_image_advGAN = torch.clamp(perturbed_image_advGAN, 0, 1)
    # adv_output_advGAN = model(perturbed_image_advGAN)
    # perturbed_image_advGAN = perturbed_image_advGAN.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    # noise_advGAN = noise_advGAN.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    # perturbed_image_advGAN = draw(perturbed_image_advGAN, adv_output_advGAN.item(), output.item())
    # perturbed_image_advGAN = imresize(perturbed_image_advGAN, (128, 128))
    # # advGAN_U
    # advGAN_uni_generator.load_state_dict(torch.load(
    #     './models/' + model_name + '_universal_netG_epoch_60.pth'))
    # noise_seed = np.load(model_name + '_noise_seed.npy')
    # noise_advGAN_U = advGAN_uni_generator(torch.from_numpy(
    #     noise_seed).type(torch.FloatTensor).to(device))
    # perturbed_image_advGAN_U = image + torch.clamp(noise_advGAN_U, -0.3, 0.3)
    # perturbed_image_advGAN_U = torch.clamp(perturbed_image_advGAN_U, 0, 1)
    # adv_output_advGAN_U = model(perturbed_image_advGAN_U)
    # perturbed_image_advGAN_U = perturbed_image_advGAN_U.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    # noise_advGAN_U = noise_advGAN_U.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    # perturbed_image_advGAN_U = draw(perturbed_image_advGAN_U, adv_output_advGAN_U.item(), output.item())
    # perturbed_image_advGAN_U = imresize(perturbed_image_advGAN_U, (128, 128))

    # for i, sample in enumerate(test_dataset):
    # batch_size = sample['image'].size(0)
    # noise_seed = np.load(model_name + '_noise_seed.npy')
    # noise_seed = np.tile(noise_seed, (batch_size, 1, 1, 1))
    # noise = advGAN_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))

    dataset_path = '../udacity-data'
    test_composed = transforms.Compose(
        [Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
    test_dataset = UdacityDataset(
        dataset_path, ['testing'], test_composed, 'test')
    test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # t0 = time.time()

    # for _, sample_batched in enumerate(test_generator):
    #     batch_x = sample_batched['image']
    #     # print(batch_x.size())
    #     # print(batch_x.size())

    #     # print(batch_y)

    #     batch_x = batch_x.type(torch.FloatTensor)

    #     batch_x = batch_x.to(device)

    #     output = model(batch_x)

    # t1 = time.time()
    # print("Total time running: %s seconds for normal prediction" %
    #       (str((t1-t0) / 5614)))

    # noise_optu = np.load(model_name + '_universal_attack_noise.npy')
    # noise_optu = torch.from_numpy(noise_optu).type(
    #     torch.FloatTensor).to(device)
    # t0 = time.time()      
    # for _, sample_batched in enumerate(test_generator):
    #     batch_x = sample_batched['image']
    #     # print(batch_x.size())
    #     # print(batch_x.size())

    #     # print(batch_y)

    #     batch_x = batch_x.type(torch.FloatTensor)

    #     batch_x = batch_x.to(device)
    #     adv_x = batch_x + noise_optu
    #     output = model(batch_x)

    # t1 = time.time()
    # print("Total time running: %s seconds for optu prediction" %
    #       (str((t1-t0) / 5614)))


    # advGAN_generator = Generator(3, 3, model_name).to(device)
    # advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))
    # advGAN_generator.eval()
    # t0 = time.time()      
    # for _, sample_batched in enumerate(test_generator):
    #     batch_x = sample_batched['image']
    #     # print(batch_x.size())
    #     # print(batch_x.size())

    #     # print(batch_y)

    #     batch_x = batch_x.type(torch.FloatTensor)

    #     batch_x = batch_x.to(device)
    #     noise = advGAN_generator(batch_x)
    #     perturbed_image = batch_x + torch.clamp(noise, -0.3, 0.3)
    #     perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #     adv_output = model(perturbed_image)

    # t1 = time.time()
    # print("Total time running: %s seconds for advGAN prediction" %
    #       (str((t1-t0) / 5614)))

    # advGAN_uni_generator = Generator(3, 3, model_name).to(device)
    # advGAN_uni_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))
    # advGAN_uni_generator.eval()
    # noise_seed = np.load(model_name + '_noise_seed.npy')
    # noise = advGAN_uni_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))

    # t0 = time.time()      
    # for _, sample_batched in enumerate(test_generator):
    #     batch_x = sample_batched['image']
    #     # print(batch_x.size())
    #     # print(batch_x.size())

    #     # print(batch_y)

    #     batch_x = batch_x.type(torch.FloatTensor)

    #     batch_x = batch_x.to(device)

    #     perturbed_image = batch_x + torch.clamp(noise, -0.3, 0.3)
    #     perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #     adv_output = model(perturbed_image)

    # t1 = time.time()
    # print("Total time running: %s seconds for advGAN_uni prediction" %
    #       (str((t1-t0) / 5614)))  

    t0 = time.time()      
    for _, sample_batched in enumerate(test_generator):
        batch_x = sample_batched['image']
        # print(batch_x.size())
        # print(batch_x.size())

        # print(batch_y)

        batch_x = batch_x.type(torch.FloatTensor)

        batch_x = batch_x.to(device)
        _, adv_x, _, _, _ = fgsm_attack(model, batch_x, target, device)
        output = model(adv_x)

    t1 = time.time()
    print("Total time running: %s seconds for fgsm prediction" %
          (str((t1-t0) / 5614)))

    # t0 = time.time()      
    # for _, sample_batched in enumerate(test_generator):
    #     batch_x = sample_batched['image']
    #     # print(batch_x.size())
    #     # print(batch_x.size())

    #     # print(batch_y)

    #     batch_x = batch_x.type(torch.FloatTensor)

    #     batch_x = batch_x.to(device)
    #     adv_x, _, _, _ = optimized_attack(model, target, batch_x, device)
    #     output = model(adv_x)

    # t1 = time.time()
    # print("Total time running: %s seconds for opt prediction" %
    #       (str((t1-t0) / 5614)))