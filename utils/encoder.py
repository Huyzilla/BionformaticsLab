import torch
import torch.nn as nn
from utils.conv_block import conv_block

class encoder(nn.Module):
    def __init__(self, in_chanels, num_filters):
        super(encoder,self).__init__()
        self.conv = conv_block(in_chanels, num_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x,p