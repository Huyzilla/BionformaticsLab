import torch
import torch.nn as nn
from utils.conv_block import conv_block
from utils.decoder import decoder
from utils.encoder import encoder

class MyUnet(nn.Module):
    def __init__(self, in_chanels, num_classes):
        super(MyUnet,self).__init__()

        self.en1 = encoder(in_chanels, 64)
        self.en2 = encoder(64, 128)
        self.en3 = encoder(128, 256)
        self.en4 = encoder(256, 512)

        self.bridge = conv_block(512, 1024)

        self.de1 = decoder(1024, 512)
        self.de2 = decoder(512, 256)
        self.de3 = decoder(256, 128)
        self.de4 = decoder(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self,inputs):
        en1, p1 = self.en1(inputs)
        # print(en1.shape)
        en2, p2 = self.en2(p1)
        # print(en2.shape)
        en3, p3 = self.en3(p2)
        # print(en3.shape)
        en4, p4 = self.en4(p3)
        # print(en4.shape)
        b = self.bridge(p4)
        # print(b.shape)
        de1 = self.de1(b,en4)
        # print(de1.shape)
        de2 = self.de2(de1,en3)
        # print(de2.shape)
        de3 = self.de3(de2,en2)
        # print(de3.shape)
        de4 = self.de4(de3,en1)
        # print(de4.shape)
        output = self.final_conv(de4)
        # print(output.shape)
        
        return output

