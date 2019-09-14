import torch.nn as nn
import torch
import numpy as np
from . import models
import torchvision
import os
# np.random.seed(0)
# torch.manual_seed(0)
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
# from data import UdacityDataset, Rescale, Preprocess, ToTensor
# from model import BaseCNN
# from torchvision import datasets, transforms
models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_name,
                #  model_num_labels,
                target,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.target = target
        # self.model_num_labels = model_num_labels
        self.model = model
        self.model_name = model_name
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc, self.model_name).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, y):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            pred_steer = self.model(adv_images)
            loss_adv = F.mse_loss(pred_steer, y)


            adv_lambda = 500
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()
        
        # print(loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item())
        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()*loss_adv.item()

    def train(self, train_dataset, epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)

        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for _, data in enumerate(train_dataloader, start=0):
                images = data['image']
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                steers = self.model(images)
                target_steers = steers + self.target
                target_steers = target_steers.type(torch.FloatTensor)
                target_steers = target_steers.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, target_steers)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            # print("epoch %d:\nloss_D: %.4f, loss_G_fake: %.4f,\
            # \nloss_perturb: %.4f, loss_adv: %.4f, \n" %
            #      (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
            #       loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%60==0:
                netG_file_name = models_path + self.model_name + '_netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)


if __name__ == "__main__":
    # root_dir = '../udacity-data'
    # target = 0.3
    # device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # target_models = []
    # basecnn = 'baseline.pt'
    # model1 = BaseCNN()
    # model1.to(device)
    # model1.load_state_dict(torch.load(basecnn))
    # model1.eval()   
    # model_name = 'baseline'
    # test_composed = transforms.Compose([Rescale((128, 128)), Preprocess(), ToTensor()])
    # train_dataset = UdacityDataset(root_dir, ['HMB1', 'HMB2', 'HMB4', 'HMB5', 'HMB6'], test_composed, type_='train')
    # advGAN = AdvGAN_Attack(
    #                 device,
    #                 model1,
    #                 model_name,
    #                 target,
    #                 3,
    #                 0,
    #                 1)
    # advGAN.train(train_dataset,)
    pass