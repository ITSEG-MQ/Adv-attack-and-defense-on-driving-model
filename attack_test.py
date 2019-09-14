import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data import UdacityDataset, Rescale, Preprocess, ToTensor
from model import BaseCNN
from viewer import draw
from scipy.misc import imresize
from torchvision import datasets, transforms
from fgsm_attack import fgsm_attack
from advGAN.models import Generator
from optimization_attack import optimized_attack
from scipy.misc import imread, imresize


def exp1_fig():
    model = BaseCNN()
    model_name = 'baseline'
    model.load_state_dict(torch.load('baseline.pt'))
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    model.eval()
    target = 0.3
    image = imread(
        'F:\\udacity-data\\testing\\center\\1479425441182877835.jpg')[200:, :]
    image = imresize(image, (128, 128))
    image = image / 255.
    image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
    image =image.type(torch.FloatTensor)
    image = image.to(device)
    output = model(image)
    print(output)

    advGAN_generator = Generator(3, 3, model_name).to(device)
    advGAN_uni_generator = Generator(3, 3, model_name).to(device)
    
    # fgsm

    _, perturbed_image_fgsm, _, adv_output_fgsm, noise_fgsm = fgsm_attack(model, image, target, device)
    print('fgsm', adv_output_fgsm)
    perturbed_image_fgsm = perturbed_image_fgsm.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    noise_fgsm = noise_fgsm.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    perturbed_image_fgsm = draw(perturbed_image_fgsm, adv_output_fgsm.item(), output.item())
    perturbed_image_fgsm = imresize(perturbed_image_fgsm, (128, 128))
    # opt
    perturbed_image_opt, noise_opt, _, adv_output_opt = optimized_attack(
        model, target, image, device)
    perturbed_image_opt = perturbed_image_opt.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    print('opt', adv_output_opt)

    noise_opt = noise_opt.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    perturbed_image_opt = draw(perturbed_image_opt, adv_output_opt.item(), output.item())
    perturbed_image_opt = imresize(perturbed_image_opt, (128, 128))
    # optu
    noise_optu = np.load(model_name + '_universal_attack_noise.npy')
    noise_optu = torch.from_numpy(noise_optu).type(torch.FloatTensor).to(device)
    perturbed_image_optu = image + noise_optu
    perturbed_image_optu = torch.clamp(perturbed_image_optu, 0, 1)
    adv_output_optu = model(perturbed_image_optu)
    print('universal', adv_output_optu)    
    perturbed_image_optu = perturbed_image_optu.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    noise_optu = noise_optu.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    perturbed_image_optu = draw(perturbed_image_optu, adv_output_optu.item(), output.item())
    perturbed_image_optu = imresize(perturbed_image_optu, (128, 128))
    # advGAN
    advGAN_generator.load_state_dict(torch.load(
        './models/' + model_name + '_netG_epoch_60.pth'))
    noise_advGAN = advGAN_generator(image)
    perturbed_image_advGAN = image + torch.clamp(noise_advGAN, -0.3, 0.3)
    perturbed_image_advGAN = torch.clamp(perturbed_image_advGAN, 0, 1)
    adv_output_advGAN = model(perturbed_image_advGAN)
    print('advGAN', adv_output_advGAN)
    perturbed_image_advGAN = perturbed_image_advGAN.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    noise_advGAN = noise_advGAN.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    perturbed_image_advGAN = draw(perturbed_image_advGAN, adv_output_advGAN.item(), output.item())
    perturbed_image_advGAN = imresize(perturbed_image_advGAN, (128, 128))
    # advGAN_U
    advGAN_uni_generator.load_state_dict(torch.load(
        './models/' + model_name + '_universal_netG_epoch_60.pth'))
    noise_seed = np.load(model_name + '_noise_seed.npy')
    noise_advGAN_U = advGAN_uni_generator(torch.from_numpy(
        noise_seed).type(torch.FloatTensor).to(device))
    perturbed_image_advGAN_U = image + torch.clamp(noise_advGAN_U, -0.3, 0.3)
    perturbed_image_advGAN_U = torch.clamp(perturbed_image_advGAN_U, 0, 1)
    adv_output_advGAN_U = model(perturbed_image_advGAN_U)    
    print('advGAN_uni', adv_output_advGAN_U)    
    perturbed_image_advGAN_U = perturbed_image_advGAN_U.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    noise_advGAN_U = noise_advGAN_U.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    perturbed_image_advGAN_U = draw(perturbed_image_advGAN_U, adv_output_advGAN_U.item(), output.item())
    perturbed_image_advGAN_U = imresize(perturbed_image_advGAN_U, (128, 128))

    plt.subplot(2,5,1)
    plt.imshow(perturbed_image_fgsm)
    # plt.text(0.3, 0.3, 'y: %.4f' % output.item())
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,2)
    plt.imshow(perturbed_image_opt)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,3)
    plt.imshow(perturbed_image_optu)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,4)
    plt.imshow(perturbed_image_advGAN)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,5)
    plt.imshow(perturbed_image_advGAN_U)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,6)
    plt.imshow(np.clip(noise_fgsm*5,0,1))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,7)
    plt.imshow(np.clip(noise_opt*5,0,1))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,8)
    plt.imshow(np.clip(noise_optu*5,0,1))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,9)
    plt.imshow(np.clip(noise_advGAN*5,0,1))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,5,10)
    plt.imshow(np.clip(noise_advGAN_U*5,0,1))
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    plt.show()
def generate_image(X, X_adv, noise, y_pred, y_adv, image_size):
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax1.title.set_text('original pred: %.4f' % y_pred)
    X = draw(X * 255, np.array([y_pred]))
    X = imresize(X, image_size)
    plt.imshow(X)
    ax2 = plt.subplot(1, 3, 2)
    ax2.title.set_text('adv pred: %.4f' % y_adv)
    X_adv = draw(X_adv * 255, np.array([y_pred]), np.array([y_adv]))
    X_adv = imresize(X_adv, image_size)
    plt.imshow(X_adv)
    ax3 = plt.subplot(1, 3, 3)
    ax3.title.set_text('5 * noise')
    plt.imshow(np.clip(noise * 5, 0, 1))
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    # plt.savefig(path.join('compare', 'TFGSM' , str(i) + '.jpg'))
    return plt


def fgsm_attack_test(model, image, target, device, epsilon=0.01, image_size=(128, 128)):
    # image = image.unsqueeze(0)
    image =image.type(torch.FloatTensor)
    image = image.to(device)
    # steer = model(image)
    # perturbed_image = image.clone()
    # # steer = steer.type(torch.FloatTensor)
    # # if (steer.item() > -0.1):
    # #     target_steer = steer + target
    # # else:
    # #     target_steer = steer - target
    # target_steer = steer - target
    # target_steer = target_steer.to(device)
    # image.requires_grad = True
    # output = model(image)
    # adv_output = output.clone()
    # diff = 0
    # # while abs(diff) < abs(target):
    # for i in range(5):
    #     loss = F.mse_loss(adv_output, target_steer)
    #     model.zero_grad()
    #     loss.backward(retain_graph=True)
    #     image_grad = image.grad.data
    #     perturbed_image = fgsm_attack(perturbed_image, epsilon, image_grad)
    #     adv_output = model(perturbed_image)
    #     diff = abs(adv_output.detach().cpu().numpy() - output.detach().cpu().numpy())
    # # diff, perturbed_image, steer, adv_output = fgsm_attack(model, image, target, device, epsilon=epsilon, image_size=image_size)
    # noise = torch.clamp(perturbed_image - image, 0, 1)
    diff, perturbed_image, steer, adv_output, noise = fgsm_attack(
        model, image, target, device)
    plt = generate_image(image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[
                         0, :, :, :], noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], steer.detach().cpu().numpy()[0][0], adv_output.detach().cpu().numpy()[0][0], image_size)
    return diff, plt, np.sum(noise.detach().cpu().numpy()), perturbed_image.detach().cpu().numpy()


def optimized_attack_test(model, image, target, device, image_size=(128, 128)):
    #image = image.unsqueeze(0)
    # steer = sample['steer']
    image = image.type(torch.FloatTensor)
    image = image.to(device)
    perturbed_image, noise, steer, adv_output = optimized_attack(
        model, target, image, device)
    diff = abs(steer.item() - adv_output.item())
    # plt = generate_image(image.squeeze().detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,:], perturbed_image.squeeze().detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,:], noise.squeeze().detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,:], steer.detach().cpu().numpy()[0][0], adv_output.detach().cpu().numpy()[0][0], (128, 128))
    plt = generate_image(image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], perturbed_image.detach().cpu().numpy().transpose(
        0, 2, 3, 1)[0, :, :, :], noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], steer.item(), adv_output.item(), image_size)
    # plt.show()
    return diff, plt, perturbed_image.detach().cpu().numpy()


def advGAN_test(model, image, advGAN_generator, device, image_size=(128, 128)):

    #image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor)
    image = image.to(device)
    steer = model(image)
    noise = advGAN_generator(image)
    perturbed_image = image + torch.clamp(noise, -0.3, 0.3)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adv_output = model(perturbed_image)
    plt = generate_image(image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[
                         0, :, :, :], noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], steer.detach().cpu().numpy()[0][0], adv_output.detach().cpu().numpy()[0][0], image_size)

    diff = abs(adv_output.detach().cpu().numpy() -
               steer.detach().cpu().numpy())
    #diff = abs(adv_output.item() - steer.item())
    return diff, plt, perturbed_image.detach().cpu().numpy()


def advGAN_uni_test(model, image, device, noise, image_size=(128, 128)):
    # image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor).to(device)
    steer = model(image)
    perturbed_image = image + torch.clamp(noise, -0.3, 0.3)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adv_output = model(perturbed_image)
    plt = generate_image(image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[
                         0, :, :, :], noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], steer.detach().cpu().numpy()[0][0], adv_output.detach().cpu().numpy()[0][0], image_size)
    # plt = generate_image(image.squeeze().detach().cpu().numpy().transpose(1, 2, 0), perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0), noise.squeeze().detach().cpu().numpy().transpose(1, 2, 0), steer.item(), adv_output.item(), (128, 128))
    diff = abs(adv_output.detach().cpu().numpy() -
               steer.detach().cpu().numpy())
    return diff, plt, perturbed_image.detach().cpu().numpy()


def optimized_uni_test(model, image, device, noise, image_size=(128, 128)):
    # image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor).to(device)
    steer = model(image)
    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adv_output = model(perturbed_image)
    plt = generate_image(image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[
                         0, :, :, :], noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, :], steer.detach().cpu().numpy()[0][0], adv_output.detach().cpu().numpy()[0][0], image_size)
    # plt = generate_image(image.squeeze().detach().cpu().numpy().transpose(1, 2, 0), perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0), noise.squeeze().detach().cpu().numpy().transpose(1, 2, 0), steer.item(), adv_output.item(), (128, 128))
    diff = abs(adv_output.detach().cpu().numpy() -
               steer.detach().cpu().numpy())
    return diff, plt, perturbed_image.detach().cpu().numpy()


if __name__ == "__main__":
    # target_model = 'cnn.pt'
    # target = 0.3
    # root_dir = '/media/dylan/Program/cg23-dataset'
    # test_composed = transforms.Compose([Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
    # full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')
    # train_size = int(0.8*len(full_dataset))
    # test_size =len(full_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # # dataloader = torch.utils.data.DataLoader(dataset,1,True)
    # model = BaseCNN()
    # model.to(device)
    # model.load_state_dict(torch.load(target_model))
    # model.eval()
    # advGAN_uni_test(model, test_dataset, target)
    # advGAN_test(model, test_dataset, target)
    # optimized_attack_test(model, train_dataset, target, device)
    # fgsm_attack_test(model, train_dataset, target, device)
    exp1_fig()
