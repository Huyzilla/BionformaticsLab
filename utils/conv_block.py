import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_chanels, num_filters):
        super(conv_block,self).__init__()
        self.conv1 = nn.Conv2d(in_chanels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x