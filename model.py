# steer angle prediction model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math

class BaseCNN(nn.Module):
    def __init__(self,type_='teacher'):
        super(BaseCNN, self).__init__()
        self.type = type_
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),

            nn.Dropout(0.25)

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),

            nn.Dropout(0.25)

        )
        self.layer4 = nn.Sequential(
            nn.Linear(16*16*128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer4(out)
        out2 = self.layer5(out)
        if self.type == 'teacher':
            return out2
        else:
            return out2, out

class Nvidia(nn.Module):
    def __init__(self, type_='teacher'):
        super(Nvidia, self).__init__()
        self.type = type_
        # 3*66*200
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 24*31*98
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 36*14*47
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 48*5*22
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 64*3*20
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 64*1*8
        self.layer6 = nn.Sequential(
            #nn.Linear(64*1*18, 1164),
            nn.Linear(64*9*9, 1164),

            nn.ReLU(inplace=True),
            nn.Linear(1164, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Linear(10, 1)
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer6(out)
        out2 = self.layer7(out)
        if self.type == 'teacher':
            return out2
        else:
            return out2, out

class Vgg16(nn.Module):
    def __init__(self, pretrained=False, type_='teacher'):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        self.type = type_
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.clf_layer1 = nn.Sequential(
            nn.BatchNorm1d(25088),
            nn.Linear(25088, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        #self.clf_layer1 = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.clf_layer2 = nn.Linear(1024, 1)

    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.clf_layer1(x)
        out2 = self.clf_layer2(out)

        if self.type == 'teacher':
            return out2
        else:
            return out2, out


def build_vgg16(pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    if pretrained:
        for parma in model.parameters():
            parma.requires_grad = False    
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(25088),
        nn.Linear(25088, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )
    return model

def weight_init(m):
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
# m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# net = BaseCNN()
# net = net.to(device)
# summary(net, (3,128,128))
