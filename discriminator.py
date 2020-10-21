import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        # input 1*64*64
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, True))

        # input 64*32*32
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.7))
        # input 128*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.6))
        # input 256*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        self.ValidityLayer=nn.Sequential(nn.Conv2d(512,1,4,1,0,bias=False),
                            nn.Sigmoid())
        self.LabelLayer=nn.Sequential(nn.Conv2d(512,10,4,1,0,bias=False),
        nn.LogSoftmax(dim=1))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.ValidityLayer(x)
        plabel = self.LabelLayer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, 10)

        return validity, plabel