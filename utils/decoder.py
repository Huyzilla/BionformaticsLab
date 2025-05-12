import torch
import torch.nn as nn
from utils.conv_block import conv_block

class decoder(nn.Module):
    def __init__(self, in_chanels, num_filters):
        super(decoder,self).__init__()
        self.up = nn.ConvTranspose2d(in_chanels, in_chanels//2, kernel_size=2, stride=2)
        self.conv = conv_block(in_chanels, num_filters)

    def forward(self,inputs,skip):
        x = self.up(inputs)
        # print(x.shape)
        x = torch.cat((x,skip),1)
        # print(x.shape)
        x = self.conv(x)
        return x