import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 model_name
                 ):
        super(Generator, self).__init__()
        self.model_name = model_name

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]
        if self.model_name == 'nvidia':

            decoder_lis = [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 11 x 11
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 23 x 23
                nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
                nn.Tanh()
                # state size. image_nc x 28 x 28
            ]
        else:
            decoder_lis = [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 11 x 11
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 23 x 23
                nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
                nn.Tanh()
                # state size. image_nc x 28 x 28
            ]
        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Uan_generator(nn.Module):
    ''' The attacker model is given an image and outputs a perturbed version of that image.''' 
    def __init__(self, imageSize):
        super(Uan_generator, self).__init__()
        self.imageSize = imageSize
        self.conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
            #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 ),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=False),
            nn.BatchNorm2d(3 ),
            nn.ReLU(True),
        )
        #self.fc = nn.Sequential(
        #    nn.Linear(3*33*33, 1024),
        #    nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
        #    nn.Linear(1024, 3*299*299),
        #)
        self.fc = nn.Sequential(
            nn.Linear(3*33*33, 512),
            nn.BatchNorm1d(512 ),
            nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024 ),
            nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
            nn.Linear(1024, 3*self.imageSize[0]*self.imageSize[1]),
        )
        self.tanh = nn.Sequential(
            nn.Tanh(),
        )
    def forward(self, noise):
        x = self.conv(noise)
        x = x.view(-1, 3*33*33)
        x = self.fc(x)
        x = x.view(-1, 3, self.imageSize[0], self.imageSize[1])
        return x

# device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# net = Generator(3,3, 'baseline')
# net = net.to(device)
# summary(net, (3,224,224))