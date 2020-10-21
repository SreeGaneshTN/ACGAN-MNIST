import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        #input 100*1*1
        self.layer1=nn.Sequential(
            nn.ConvTranspose2d(100,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        # input 256*8*8
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        # input 128*16*16
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        # input 64*32*32
        self.layer5 = nn.Sequential(
        nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
        nn.Tanh()
        )

        # output 1*64*64

        self.embedding = nn.Embedding(10, 100)

    def forward(self,x,labels):
        emb=self.embedding(labels)
        x = x.view(-1, 100, 1, 1)
        x=torch.mul(x,emb)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)

        return x
